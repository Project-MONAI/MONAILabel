"""Shared discovery, verified archive installation, and atomic cache publication."""

import glob
import hashlib
import json
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import httpx
from filelock import FileLock

from monailabel.core.errors import DomainError
from monailabel.viewers.provisioning.catalog import Installation, ToolSpec, catalog, release


def executable_file(path: Path, system: str | None = None) -> bool:
    return path.is_file() and (
        (system or platform.system()) == "Windows" or os.access(path, os.X_OK)
    )


def safe_path(name: str) -> None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or PureWindowsPath(name).drive or ":" in name:
        raise DomainError("Viewer archive contains an unsafe path.")


def extract(archive: Path, destination: Path, kind: str) -> None:
    if kind == "zip":
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if sum(m.file_size for m in members) > 12 * 1024**3 or len(members) > 200000:
                raise DomainError("Viewer archive exceeds the extraction size limit.")
            for member in members:
                safe_path(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise DomainError("Viewer ZIP archive contains a symbolic link.")
            bundle.extractall(destination)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            entries = bundle.getmembers()
            if sum(m.size for m in entries) > 12 * 1024**3 or len(entries) > 200000:
                raise DomainError("Viewer archive exceeds the extraction size limit.")
            for entry in entries:
                safe_path(entry.name)
            bundle.extractall(destination, filter="data")


class DesktopProvisioner:
    def __init__(self, root: Path):
        self.root = root

    def discover(self, viewer: str) -> Path | None:
        variable = f"MONAILABEL_{viewer.upper()}_EXECUTABLE"
        if configured := os.environ.get(variable):
            path = Path(configured).expanduser()
            if not executable_file(path):
                raise DomainError(f"{variable} is not an executable file.")
            return path.resolve()
        name = "QuPath" if viewer == "qupath" else "Slicer"
        if found := shutil.which(name) or shutil.which(name + ".exe"):
            return Path(found).resolve()
        if (
            viewer == "slicer"
            and platform.system() == "Linux"
            and platform.machine() in {"aarch64", "arm64"}
        ):
            # The optional source builder publishes here atomically. Do not
            # start an hours-long compilation implicitly from a viewer click.
            built = list((self.root / "slicer/native-install").glob("*/Slicer"))
            if len(built) == 1 and executable_file(built[0]):
                return built[0].resolve()
        patterns = catalog().discovery.get(viewer, {}).get(platform.system(), [])
        for pattern in patterns:
            paths = sorted(glob.glob(os.path.expandvars(os.path.expanduser(pattern))), reverse=True)
            for candidate in paths:
                if executable_file(Path(candidate)) and "(console)" not in candidate:
                    return Path(candidate).resolve()
        return None

    def ensure(
        self,
        viewer: str,
        *,
        spec: ToolSpec | None = None,
        download: bool = True,
        progress: Callable[[str], None] = print,
    ) -> Installation:
        if existing := self.discover(viewer):
            progress(f"Reusing installed {viewer}: {existing}")
            return Installation(viewer, str(existing), "system", "existing")
        spec = spec or release(viewer)
        if spec is None:
            raise DomainError(
                f"No automatic {viewer} installation recipe for {platform.system()} "
                f"{platform.machine()}. Set MONAILABEL_{viewer.upper()}_EXECUTABLE "
                "to an existing installation."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / spec.name / f"{spec.version}-{spec.system}-{spec.machine}"
        with FileLock(str(self.root / f"{spec.name}.lock"), timeout=1800):
            manifest = destination / "installation.json"
            if manifest.exists():
                metadata = json.loads(manifest.read_text())
                path = (destination / metadata["executable"]).resolve()
                if path.is_relative_to(destination.resolve()) and executable_file(
                    path, spec.system
                ):
                    progress(f"Reusing cached {viewer}: {path}")
                    return Installation(viewer, str(path), "cache", spec.version)
            if not download:
                raise DomainError(
                    f"{viewer} is not installed or cached. Enable download to install it."
                )
            if spec.system != platform.system():
                raise DomainError("The installation recipe does not match this operating system.")
            progress(f"Downloading verified {viewer} {spec.version} to {destination} ...")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".install-") as temp:
                staging = Path(temp)
                archive = staging / "download"
                digest = hashlib.new(spec.algorithm)
                with httpx.stream("GET", spec.url, follow_redirects=True, timeout=60) as response:
                    response.raise_for_status()
                    with archive.open("wb") as output:
                        total = 0
                        for block in response.iter_bytes(1024 * 1024):
                            total += len(block)
                            if total > 4 * 1024**3:
                                raise DomainError(
                                    "Viewer download exceeds the installation size limit."
                                )
                            output.write(block)
                            digest.update(block)
                if digest.hexdigest() != spec.checksum:
                    raise DomainError("Viewer checksum mismatch. The download was discarded.")
                unpacked = staging / "unpacked"
                unpacked.mkdir()
                extract(archive, unpacked, spec.archive)
                if spec.build == "qupath":
                    from monailabel.viewers.provisioning.qupath_build import build_qupath

                    unpacked = build_qupath(unpacked, self.root, progress)
                candidates = sorted(unpacked.glob(spec.executable))
                if len(candidates) != 1 or not executable_file(candidates[0], spec.system):
                    raise DomainError("Downloaded viewer does not contain the expected executable.")
                relative = candidates[0].relative_to(unpacked)
                (unpacked / "installation.json").write_text(
                    json.dumps({**asdict(spec), "executable": relative.as_posix()}, indent=2) + "\n"
                )
                if destination.exists():
                    raise DomainError("Cached installation is incomplete. Move it aside and retry.")
                unpacked.rename(destination)
            return Installation(viewer, str(destination / relative), "download", spec.version)
