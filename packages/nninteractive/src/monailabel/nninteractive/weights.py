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

"""Immutable official model download; verify every file before loading the checkpoint."""

import hashlib
import os
from pathlib import Path

import httpx
from filelock import FileLock
from platformdirs import user_cache_path

from monailabel.core.errors import DomainError
from monailabel.core.ports import Progress

REVISION = "3f308d751c00644e4fde6f09c600264b393b21b5"
FILES = {
    "LICENSE": (67, "4f60f5747c5506020923866690c2a41a3c74ffa85b7371eac2b02e23185f91d5"),
    "dataset.json": (245, "e090394ac539b2823b2135173658190245c462cb042d3442487e4829f08252c4"),
    "fold_0/checkpoint_final.pth": (
        411387150,
        "b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9",
    ),
    "inference_session_class.json": (
        121,
        "8f43587a9d139fcf7af8776219e9432dab079f985cb0d7da72a4c851b39a4f09",
    ),
    "plans.json": (6287, "64a30a430438302a692a3edadd345a207b3428bea2e5cdc125298757da21f4d4"),
}


def cache_root() -> Path:
    return (
        Path(os.environ.get("MONAILABEL_MODELS_DIR", str(user_cache_path("monailabel") / "models")))
        / "nninteractive"
    )


def verify(path: Path, size: int, checksum: str) -> None:
    with path.open("rb") as stream:
        if (
            path.stat().st_size != size
            or hashlib.file_digest(stream, "sha256").hexdigest() != checksum
        ):
            raise DomainError("nnInteractive model checksum verification failed.")


def weights(progress: Progress) -> Path:
    root = cache_root() / REVISION
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "download.lock"), timeout=1800):
        for name, (size, checksum) in FILES.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            progress(0)
            if not path.exists():
                temporary = path.with_suffix(".part")
                try:
                    with (
                        httpx.stream(
                            "GET",
                            f"https://huggingface.co/MIC-DKFZ/nnInteractive/resolve/{REVISION}/"
                            f"nnInteractive_v1.0/{name}",
                            follow_redirects=True,
                            timeout=120,
                        ) as response,
                        temporary.open("wb") as output,
                    ):
                        response.raise_for_status()
                        received = 0
                        for chunk in response.iter_bytes(1024 * 1024):
                            received += len(chunk)
                            if received > size:
                                raise DomainError(
                                    "nnInteractive download exceeded its pinned size."
                                )
                            output.write(chunk)
                            progress(0.1 * received / size)
                    verify(temporary, size, checksum)
                    temporary.chmod(0o444)
                    temporary.replace(path)
                except httpx.HTTPError as exc:
                    raise DomainError("Could not download nnInteractive. Retry online.") from exc
                finally:
                    temporary.unlink(missing_ok=True)
            verify(path, size, checksum)
    return root
