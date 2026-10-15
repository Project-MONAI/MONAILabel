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

"""Cold workspace backups. Never stops services or overwrites an existing restore.

Use after saving viewer drafts and stopping every workspace writer in an approved
maintenance window. Run without --apply first. Configuration, external viewer
volumes, software and model caches need their own separately reviewed backups.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

EXCLUDED = {"server.lock", ".cache"}
HEADROOM = 1024**3


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, Path]:
    """Do not follow links or cross mounts; refuse unknown non-file state."""
    result = {}
    device = root.stat().st_dev
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        if parent == root:
            directories[:] = [name for name in directories if name not in EXCLUDED]
            files = [name for name in files if name not in EXCLUDED]
        for name in directories + files:
            path = parent / name
            info = path.lstat()
            if info.st_dev != device or stat.S_ISLNK(info.st_mode):
                raise ValueError("Linked or separately mounted data needs an explicit backup plan.")
            if stat.S_ISDIR(info.st_mode):
                continue
            if stat.S_ISSOCK(info.st_mode):
                raise ValueError(
                    "Workspace contains a socket; stop and inspect viewer writers first."
                )
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Workspace contains an unsupported special file.")
            result[path.relative_to(root).as_posix()] = path
    return result


@contextmanager
def stopped_workspace(root: Path):
    # Use the same existing inode as filelock.UnixFileLock, without truncating it.
    # Never create/unlink a lock: replacing one can allow concurrent owners.
    descriptor = os.open(root / "server.lock", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("The workspace lock is not a regular file.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Workspace is active. No backup was created.") from error
        yield
    finally:
        os.close(descriptor)


def check_workspace(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Use an existing, non-linked workspace directory.")
    for name in ("monailabel.sqlite3", "secrets.key", "server.lock"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Workspace is missing a regular {name} file.")


def destination_directory(path: Path, source: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Destination must be an existing, non-linked private directory.")
    if path.resolve().is_relative_to(source.resolve()):
        raise ValueError("Backup destination must be outside the workspace.")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Destination must be owned by this user and accessible only to them.")


def copy_checked(source: Path, target: Path) -> dict[str, str | int]:
    before = source.stat()
    # Path.mkdir(parents=True) applies mode only to the leaf directory.
    # Explicitly protect every newly created ancestor without chmod'ing old data.
    missing = []
    parent = target.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        os.chmod(target, 0o600)
        digest = hashlib.sha256()
        size = 0
        while chunk := incoming.read(1024 * 1024):
            outgoing.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    after = source.stat()
    if (
        (before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_ino, after.st_size, after.st_mtime_ns)
        or size != before.st_size
        or file_hash(target) != digest.hexdigest()
    ):
        raise ValueError("A file changed or failed verification; backup remains incomplete.")
    return {"bytes": size, "sha256": digest.hexdigest()}


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_backup(workspace: Path, destination: Path, *, apply: bool, writers_stopped: bool):
    check_workspace(workspace)
    destination_directory(destination, workspace)
    if apply and not writers_stopped:
        raise ValueError("Confirm all server, viewer and job writers are stopped first.")
    with stopped_workspace(workspace):
        files = inventory(workspace)
        size = sum(path.stat().st_size for path in files.values())
        if shutil.disk_usage(destination).free < size + HEADROOM:
            raise ValueError("Destination lacks room for this backup plus 1 GiB headroom.")
        if not apply:
            return {"mode": "plan", "files": len(files), "bytes": size, "changes": False}
        incomplete = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=destination))
        copied = incomplete / "workspace"
        copied.mkdir(mode=0o700)
        entries = {name: copy_checked(source, copied / name) for name, source in files.items()}
        if set(inventory(workspace)) != set(files):
            raise ValueError("Workspace changed during backup; backup remains incomplete.")
        manifest = {"version": 1, "excluded": sorted(EXCLUDED), "files": entries}
        with (incomplete / "manifest.json").open("x") as output:
            os.chmod(incomplete / "manifest.json", 0o600)
            json.dump(manifest, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        verify_backup(incomplete)
        for directory, _, _ in os.walk(incomplete, topdown=False):
            sync_directory(Path(directory))
        complete = destination / incomplete.name.replace(".incomplete-", "backup-", 1)
        if complete.exists():
            raise ValueError("Backup name already exists; nothing was overwritten.")
        incomplete.rename(complete)
        sync_directory(destination)
        return {"mode": "created", "backup": str(complete), "files": len(entries), "bytes": size}


def verify_backup(backup: Path) -> dict:
    if backup.is_symlink() or (backup / "manifest.json").is_symlink():
        raise ValueError("Backup and manifest must not be links.")
    manifest = json.loads((backup / "manifest.json").read_text())
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported backup manifest.")
    root = backup / "workspace"
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Backup workspace must be a directory, not a link.")
    entries = manifest["files"]
    for required in ("monailabel.sqlite3", "secrets.key"):
        if required not in entries:
            raise ValueError("Backup is missing required workspace state.")
    actual = inventory(root)
    if set(actual) != set(entries):
        raise ValueError("Backup file list differs from its manifest.")
    for name, expected in entries.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Unsafe backup path.")
        path = actual[name]
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError("Backup checksum verification failed.")
    return manifest


def restore_copy(backup: Path, target: Path) -> dict:
    manifest = verify_backup(backup)
    # Restore creates only a NEW directory; never replaces a live workspace.
    if target.exists() or target.is_symlink():
        raise ValueError("Restore target already exists; nothing was overwritten.")
    destination_directory(target.parent, backup)
    size = sum(entry["bytes"] for entry in manifest["files"].values())
    if shutil.disk_usage(target.parent).free < size + HEADROOM:
        raise ValueError("Restore target lacks free space plus 1 GiB headroom.")
    target.mkdir(mode=0o700)
    for name, expected in manifest["files"].items():
        if copy_checked(backup / "workspace" / name, target / name) != expected:
            raise ValueError("Backup changed during restoration; target is incomplete.")
    return {"mode": "restored_copy", "files": len(manifest["files"]), "bytes": size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("--workspace", required=True, type=Path)
    create.add_argument("--destination", required=True, type=Path)
    create.add_argument("--apply", action="store_true")
    create.add_argument("--writers-stopped", action="store_true")
    verify = actions.add_parser("verify")
    verify.add_argument("backup", type=Path)
    restore = actions.add_parser("restore-copy")
    restore.add_argument("backup", type=Path)
    restore.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "create":
            result = create_backup(
                args.workspace,
                args.destination,
                apply=args.apply,
                writers_stopped=args.writers_stopped,
            )
        elif args.action == "restore-copy":
            result = restore_copy(args.backup, args.target)
        else:
            manifest = verify_backup(args.backup)
            result = {"mode": "verified", "files": len(manifest["files"])}
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Backup refused: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
