"""Read shared volume files without depending on the server's storage layout."""

import os
import shutil
from pathlib import Path


def volume_name(filename: str) -> str:
    name = Path(filename).name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or "Image"
    return name


def local_nifti(source: str | None, directory: Path) -> Path | None:
    """Give an immutable source a NIfTI suffix in the viewer's private session folder.

    Copy locally without downloading or decompressing, then reuse within the session.
    Keep a separate file so native viewer saves cannot modify the immutable source.
    Unavailable files fall back to authenticated download.
    """
    if not source:
        return None
    original = Path(source)
    try:
        with original.open("rb") as stream:
            suffix = ".nii.gz" if stream.read(2) == b"\x1f\x8b" else ".nii"
        target = directory / (original.name + suffix)
        if target.is_file():
            return target
        temporary = target.with_name(target.name + ".partial")
        try:
            shutil.copyfile(original, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
    except OSError:
        return None
