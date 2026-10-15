# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional, host-prepared base predictions. Inference never writes this cache."""

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from monailabel.core.errors import DomainError
from monailabel.core.ports import Mask, Volume
from monailabel.totalsegmentator.catalog import VERSION, targets
from monailabel.totalsegmentator.config import TotalConfig
from monailabel.totalsegmentator.weights import manifest


def identity(provider: str, volume: Volume, config: TotalConfig) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "format": 1,
                "provider": provider,
                "version": VERSION,
                "weights": manifest(provider)["sha256"],
                "fast": True,
                "shape": volume.image.shape,
                "affine": volume.affine,
                "config": config.model_dump(mode="json"),
            },
            sort_keys=True,
        ).encode()
    )
    # Bound the extra contiguous copy to one plane, including Fortran-order NIfTI arrays.
    for plane in volume.image:
        digest.update(np.ascontiguousarray(plane, dtype="<f4").tobytes())
    return digest.hexdigest()


def read(root: Path, key: str, provider: str, shape: tuple[int, ...]) -> Mask | None:
    report = root / key / "prediction.json"
    if not report.is_file():
        return None
    try:
        metadata = json.loads(report.read_text())
        path = root / key / "mask.npy"
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if metadata["identity"] != key or metadata["mask_sha256"] != checksum:
            raise ValueError("Checksum mismatch")
        values = np.load(path, allow_pickle=False)
        if (
            values.dtype != np.uint8
            or values.shape != shape
            or not set(np.unique(values)) <= {0, *targets(provider).values()}
        ):
            raise ValueError("Invalid geometry or labels")
        return cast(Mask, values)
    except (OSError, ValueError, KeyError) as error:
        raise DomainError(
            "Prepared TotalSegmentator prediction is invalid; ask the host to rebuild it."
        ) from error
