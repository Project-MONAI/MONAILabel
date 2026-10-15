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

"""Run an unchanged private demo against a verified, already-migrated filesystem."""

import argparse
import json
import os
import stat
from pathlib import Path


def private_directory(path: Path, device: int) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or info.st_dev != device
        or path.resolve() != path
    ):
        raise ValueError(f"Expected a private directory on the bounded filesystem: {path}")


def environment(storage: Path, workspace: Path, sockets: Path) -> dict[str, str]:
    if os.geteuid() == 0:
        raise ValueError("Run as the dedicated non-root application account, never sudo.")
    if not storage.is_absolute() or storage.resolve() != storage or not storage.is_mount():
        raise ValueError("Bounded storage is not mounted; refusing unbounded fallback writes.")
    info = storage.lstat()
    if info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
        raise ValueError("The filesystem root must be administrator-owned and protected.")
    capacity = os.statvfs(storage)
    if not 128 * 1024**3 * 0.9 < capacity.f_blocks * capacity.f_frsize <= 128 * 1024**3:
        raise ValueError("The filesystem does not have the reviewed 128 GiB capacity.")
    for name in ("workspace", "tmp", "native-sockets", "logs"):
        private_directory(storage / name, info.st_dev)
    if not workspace.is_relative_to(storage / "workspace"):
        raise ValueError("The workspace must be inside the bounded workspace directory.")
    private_directory(workspace, info.st_dev)
    for name in ("monailabel.sqlite3", "secrets.key"):
        entry = (workspace / name).lstat()
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_uid != os.getuid()
            or stat.S_IMODE(entry.st_mode) & 0o077
            or entry.st_dev != info.st_dev
        ):
            raise ValueError("Expected migrated, private database and credential-key files.")
    # Keep the short alias un-resolved in the environment: Linux socket paths
    # include the desktop namespace and session ID and must fit in 107 bytes.
    if not sockets.is_absolute() or sockets.resolve() != storage / "native-sockets":
        raise ValueError("Socket alias must point to this filesystem's native-sockets directory.")
    sample = sockets / ("monailabel-desktop-" + "0" * 12) / ("0" * 36 + "-api") / "api.sock"
    if len(os.fsencode(sample)) > 107:
        raise ValueError("Use a shorter socket alias; the native viewer socket path is too long.")
    return {
        **os.environ,
        "MONAILABEL_DATA_DIR": str(workspace),
        "TMPDIR": str(storage / "tmp"),
        "TMP": str(storage / "tmp"),
        "TEMP": str(storage / "tmp"),
        "MONAILABEL_DESKTOP_SOCKET_DIR": str(sockets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--socket-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        settings = environment(args.storage, args.workspace, args.socket_dir)
        if args.check_only:
            print(
                json.dumps(
                    {
                        key: settings[key]
                        for key in (
                            "MONAILABEL_DATA_DIR",
                            "TMPDIR",
                            "MONAILABEL_DESKTOP_SOCKET_DIR",
                        )
                    },
                    indent=2,
                )
            )
            return
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("Specify an application command after --, or use --check-only.")
        os.umask(0o077)
        os.execvpe(command[0], command, settings)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Bounded demo startup refused: {error}\n")


if __name__ == "__main__":
    main()
