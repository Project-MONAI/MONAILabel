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

"""Checksum-verified shared base checkpoints, never modified by project training."""

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import httpx
from filelock import FileLock
from platformdirs import user_cache_path

from monailabel.core.errors import DomainError
from monailabel.core.ports import Progress
from monailabel.totalsegmentator.catalog import MODELS


@cache
def manifest(provider: str) -> dict[str, Any]:
    data: dict[str, dict[str, Any]] = json.loads(
        files(__package__).joinpath("resources/weights.json").read_text()
    )
    return data[MODELS[provider][1]]


def pretrained_weights(provider: str, progress: Progress = lambda _: None) -> Path:
    spec = manifest(provider)
    root = Path(os.environ.get("MONAILABEL_MODELS_DIR", user_cache_path("monailabel") / "models"))
    root = root / "totalsegmentator" / str(spec["sha256"])
    root.mkdir(parents=True, exist_ok=True)
    folder = root / str(spec["folder"])
    lock = FileLock(str(root / "download.lock"), timeout=0.25)
    from filelock import Timeout

    while True:
        progress(0)
        try:
            lock.acquire()
            break
        except Timeout:
            continue
    try:
        if not folder.exists():
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                path = Path(temporary) / "weights.zip"
                digest, size = hashlib.sha256(), 0
                with httpx.stream(
                    "GET", spec["url"], follow_redirects=True, timeout=120
                ) as response:
                    response.raise_for_status()
                    with path.open("wb") as output:
                        for chunk in response.iter_bytes(1024 * 1024):
                            size += len(chunk)
                            if size > spec["bytes"]:
                                raise DomainError(
                                    "TotalSegmentator download exceeded its pinned size."
                                )
                            digest.update(chunk)
                            output.write(chunk)
                            progress(0)
                if size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
                    raise DomainError("TotalSegmentator checkpoint checksum verification failed.")
                with zipfile.ZipFile(path) as archive:
                    # Extract only files in the verified manifest; never follow archive paths.
                    for name, checksum in spec["files"].items():
                        target = Path(temporary) / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        content = archive.read(name)
                        if hashlib.sha256(content).hexdigest() != checksum:
                            raise DomainError("TotalSegmentator checkpoint contents differ.")
                        target.write_bytes(content)
                        target.chmod(0o444)
                shutil.move(Path(temporary) / spec["folder"], folder)
        for name, checksum in spec["files"].items():
            progress(0)
            path = root / name
            if not path.is_file():
                raise DomainError("TotalSegmentator cache is incomplete. Remove it and retry.")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                    raise DomainError(
                        "Cached TotalSegmentator weights differ from the pinned release."
                    )
    finally:
        lock.release()
    return folder
