"""Pinned public dataset downloads and bounded archive reads; never extract archive paths."""

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from filelock import FileLock, Timeout
from pydantic import TypeAdapter

from monailabel.core.dataset_templates import DatasetTemplate
from monailabel.core.errors import DomainError
from monailabel.core.video import VideoImport
from monailabel.server.data import MAX_FILE_BYTES
from monailabel.server.jobs import JobContext
from monailabel.server.workspace import workspace_dir


class Source(DatasetTemplate):
    url: str
    checksum: str
    format: Literal["msd", "totalsegmentator", "image", "video", "external"]
    video: VideoImport | None = None


def sources() -> list[Source]:
    content = files("monailabel.server").joinpath("resources/dataset-templates.json").read_text()
    return TypeAdapter(list[Source]).validate_json(content)


def dataset_cache_dir(directory: Path | None = None) -> Path:
    root = Path(
        os.environ.get("MONAILABEL_CACHE_DIR", str((directory or workspace_dir()) / ".cache"))
    )
    return Path(os.environ.get("MONAILABEL_DATASETS_DIR", str(root / "datasets"))).expanduser()


class Downloads:
    def __init__(self, root: Path, legacy_root: Path | None = None):
        self.root = root
        self.legacy = (
            Downloads(legacy_root)
            if legacy_root is not None and legacy_root.resolve() != root.resolve()
            else None
        )

    def path(self, source: Source) -> Path:
        return self.root / (source.id + Path(source.url).suffix)

    def cached(self, source: Source) -> bool:
        return self._cached_here(source) or bool(self.legacy and self.legacy._cached_here(source))

    def _cached_here(self, source: Source) -> bool:
        path = self.path(source)
        try:
            metadata = json.loads(path.with_suffix(".verified.json").read_text())
            return bool(metadata == self._metadata(source))
        except (OSError, ValueError):
            return False

    def _metadata(self, source: Source) -> dict[str, str | int]:
        stat = self.path(source).stat()
        return {
            "checksum": source.checksum,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }

    def _remember(self, source: Source) -> None:
        path = self.path(source).with_suffix(".verified.json")
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(self._metadata(source)))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _verify_saved(self, source: Source, context: JobContext) -> bool:
        """Recover complete downloads after a missing marker or interrupted publication."""
        path = self.path(source)
        if not path.is_file():
            return False
        algorithm, expected = source.checksum.split(":", 1)
        digest = hashlib.new(algorithm)
        size = path.stat().st_size
        checked = 0
        context.progress(0, "Checking the saved download")
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
                checked += len(block)
                if checked % (64 * 1024**2) == 0:
                    context.progress(0.5 * checked / max(size, 1), "Checking the saved download")
        if digest.hexdigest() != expected:
            return False
        self._remember(source)
        return True

    def _reuse_legacy(self, source: Source, context: JobContext) -> bool:
        legacy = self.legacy
        if legacy is None or not legacy.path(source).is_file():
            return False
        with FileLock(str(legacy.path(source)) + ".lock", timeout=0):
            if not (legacy._cached_here(source) or legacy._verify_saved(source, context)):
                return False
            context.progress(0, "Reusing the previously downloaded archive")
            path = self.path(source)
            part = path.with_suffix(path.suffix + ".part")
            part.unlink(missing_ok=True)
            try:
                try:
                    # Same-disk adoption costs no extra archive-sized copy. Keep the
                    # old path intact for any already-open workspace import.
                    os.link(legacy.path(source), part)
                except OSError:
                    size = legacy.path(source).stat().st_size
                    if shutil.disk_usage(self.root).free < size + 512 * 1024**2:
                        raise DomainError(
                            "Not enough disk space to copy the saved dataset cache."
                        ) from None
                    copied = 0
                    with legacy.path(source).open("rb") as incoming, part.open("wb") as output:
                        while block := incoming.read(8 * 1024**2):
                            output.write(block)
                            copied += len(block)
                            if copied % (64 * 1024**2) == 0:
                                context.progress(
                                    0.5 * copied / max(size, 1), "Copying the saved dataset cache"
                                )
                part.replace(path)
                self._remember(source)
                legacy._remember(source)
            finally:
                part.unlink(missing_ok=True)
        return True

    def fetch(self, source: Source, context: JobContext) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(source)
        try:
            with FileLock(str(path) + ".lock", timeout=0):
                if (
                    self._cached_here(source)
                    or self._verify_saved(source, context)
                    or self._reuse_legacy(source, context)
                ):
                    context.progress(0.5, "Using the verified cached download")
                    return path
                if shutil.disk_usage(self.root).free < source.download_bytes + 512 * 1024**2:
                    raise DomainError("Not enough server disk space for this dataset download.")
                part = path.with_suffix(path.suffix + ".part")
                algorithm, expected = source.checksum.split(":", 1)
                digest = hashlib.new(algorithm)
                total = 0
                try:
                    with httpx.stream(
                        "GET", source.url, follow_redirects=True, timeout=30
                    ) as reply:
                        reply.raise_for_status()
                        bound = max(source.download_bytes * 1.05, 2 * 1024**2)
                        with part.open("wb") as stream:
                            for block in reply.iter_bytes(1024**2):
                                total += len(block)
                                if total > bound:
                                    raise DomainError(
                                        "Download exceeds the catalog size; update this template."
                                    )
                                context.progress(
                                    0.5 * total / max(source.download_bytes, 1),
                                    f"Downloading {total / 1024**2:.0f} of "
                                    f"{source.download_bytes / 1024**2:.0f} MiB",
                                )
                                stream.write(block)
                                digest.update(block)
                    if digest.hexdigest() != expected:
                        raise DomainError("Dataset checksum did not match; no cases were imported.")
                    part.replace(path)
                    self._remember(source)
                except httpx.HTTPError as error:
                    raise DomainError(
                        "Dataset download failed. Retry or check the source website."
                    ) from error
                finally:
                    part.unlink(missing_ok=True)
        except Timeout as error:
            raise DomainError(
                "This dataset is already downloading. Wait for that import to finish."
            ) from error
        return path


class Archive:
    def __init__(self, path: Path):
        self.file: zipfile.ZipFile | tarfile.TarFile
        if zipfile.is_zipfile(path):
            self.file = zipfile.ZipFile(path)
            entries = [(item.filename, item.is_dir()) for item in self.file.infolist()]
        else:
            self.file = tarfile.open(path)  # noqa: SIM115 — closed by Archive.close
            entries = [(item.name, not item.isfile()) for item in self.file.getmembers()]
        if len(entries) > 250000:
            self.close()
            raise DomainError("Dataset archive has too many entries.")
        self.names: dict[str, str] = {}
        for name, directory in entries:
            key = PurePosixPath(name)
            if key.is_absolute() or ".." in key.parts or "\\" in name:
                self.close()
                raise DomainError("Dataset archive contains unsafe paths.")
            if directory or any(part.startswith("._") or part == "__MACOSX" for part in key.parts):
                continue
            if str(key) in self.names:
                self.close()
                raise DomainError("Dataset archive contains duplicate filenames.")
            self.names[str(key)] = name

    def read(self, name: str, limit: int = MAX_FILE_BYTES) -> bytes:
        original = self.names.get(name)
        if original is None:
            raise DomainError(f"Dataset is missing {name}.")
        if isinstance(self.file, zipfile.ZipFile):
            info = self.file.getinfo(original)
            if info.file_size > limit:
                raise DomainError(f"{name} exceeds the per-file import limit.")
            stream = self.file.open(info)
        else:
            member = self.file.getmember(original)
            if not member.isfile() or member.size > limit:
                raise DomainError(f"{name} is not a supported bounded file.")
            extracted = self.file.extractfile(member)
            if extracted is None:
                raise DomainError(f"Cannot read {name}.")
            stream = extracted
        with stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise DomainError(f"{name} exceeds the per-file import limit.")
        return content

    def close(self) -> None:
        self.file.close()
