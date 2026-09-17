"""Verified shared weights. Training only reads this cache and writes new artifacts."""

import hashlib
import os
from functools import cache
from pathlib import Path

import httpx
from filelock import FileLock
from platformdirs import user_cache_path

from monailabel.core.errors import DomainError
from monailabel.providers.vista3d import CHECKSUM, REVISION, WEIGHT_BYTES, WEIGHT_URL


@cache
def pretrained_weights() -> Path:
    root = (
        Path(os.environ.get("MONAILABEL_MODELS_DIR", str(user_cache_path("monailabel") / "models")))
        / "vista3d"
        / REVISION
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model.pt"
    with FileLock(str(root / "download.lock"), timeout=1800):
        if not path.exists():
            temporary = root / "model.part"
            digest = hashlib.sha256()
            size = 0
            try:
                with (
                    httpx.stream("GET", WEIGHT_URL, follow_redirects=True, timeout=120) as response,
                    temporary.open("wb") as output,
                ):
                    response.raise_for_status()
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > WEIGHT_BYTES:
                            raise DomainError("VISTA3D download exceeded its pinned size.")
                        digest.update(chunk)
                        output.write(chunk)
                if size != WEIGHT_BYTES or digest.hexdigest() != CHECKSUM:
                    raise DomainError("VISTA3D checkpoint checksum verification failed.")
                temporary.chmod(0o444)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with path.open("rb") as stream:
            if (
                path.stat().st_size != WEIGHT_BYTES
                or hashlib.file_digest(stream, "sha256").hexdigest() != CHECKSUM
            ):
                raise DomainError("Cached VISTA3D weights differ from the pinned checkpoint.")
    return path
