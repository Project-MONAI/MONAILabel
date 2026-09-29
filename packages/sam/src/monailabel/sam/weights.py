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

"""Download and verify immutable shared SAM checkpoints."""

import hashlib
import os
from functools import cache
from pathlib import Path

import httpx
from filelock import FileLock
from platformdirs import user_cache_path

from monailabel.core.errors import DomainError
from monailabel.providers.sam import MODELS


@cache
def weights(provider: str) -> Path:
    spec = MODELS[provider]
    root = (
        Path(os.environ.get("MONAILABEL_MODELS_DIR", str(user_cache_path("monailabel") / "models")))
        / provider
        / spec.revision
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec.filename
    with FileLock(str(root / "download.lock"), timeout=1800):
        if not path.exists():
            temporary = root / "weights.part"
            try:
                with (
                    httpx.stream(
                        "GET",
                        f"https://huggingface.co/{spec.repository}/resolve/"
                        f"{spec.revision}/{spec.filename}",
                        follow_redirects=True,
                        timeout=120,
                    ) as response,
                    temporary.open("wb") as output,
                ):
                    response.raise_for_status()
                    size = 0
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > spec.size:
                            raise DomainError("SAM checkpoint exceeded its pinned size.")
                        output.write(chunk)
                verify(temporary, spec.size, spec.checksum)
                temporary.chmod(0o444)
                temporary.replace(path)
            except httpx.HTTPError as exc:
                raise DomainError(
                    "Could not download the pinned SAM checkpoint. Retry online."
                ) from exc
            finally:
                temporary.unlink(missing_ok=True)
        verify(path, spec.size, spec.checksum)
    return path


def verify(path: Path, size: int, checksum: str) -> None:
    with path.open("rb") as stream:
        if (
            path.stat().st_size != size
            or hashlib.file_digest(stream, "sha256").hexdigest() != checksum
        ):
            raise DomainError("SAM checkpoint checksum verification failed.")
