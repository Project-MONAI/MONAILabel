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

"""Prepare NEW bounded storage; never migrate data or change the application."""

import argparse
import json
import os
import pwd
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path("/var/lib/monailabel-demo-storage")
IMAGE = ROOT / "demo-data.ext4"
MOUNT = ROOT / "data"
UNIT_DIR = Path("/etc/systemd/system")
PROTECTED_SCRIPT = Path("/root/monailabel-demo-storage.py")
SIZE = 128 * 1024**3
HEADROOM = 32 * 1024**3
TOOLS = ("fallocate", "systemctl", "systemd-escape", "findmnt")
ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}


def run(*args: str) -> str:
    return subprocess.run(
        args, check=True, text=True, capture_output=True, env=ENV, timeout=300
    ).stdout.strip()


def absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"Already exists; nothing will be overwritten: {path}")


def protected_directory(path: Path) -> None:
    for entry in (path, *path.parents):
        info = entry.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError(f"An administrator-owned, non-writable parent is required: {entry}")


def unit_name() -> str:
    name = run("/usr/bin/systemd-escape", "--path", "--suffix=mount", str(MOUNT))
    if not name.endswith(".mount") or "/" in name or "\n" in name:
        raise ValueError("Invalid mount unit name")
    return name


def unit_text() -> str:
    return (
        "[Unit]\nDescription=Bounded persistent MONAI demo storage\n"
        "RequiresMountsFor=/var/lib\n\n"
        f"[Mount]\nWhat={IMAGE}\nWhere={MOUNT}\nType=ext4\n"
        "Options=loop,nodev,nosuid\nTimeoutSec=120\n\n"
        "[Install]\nWantedBy=multi-user.target\n"
    )


def write_new(path: Path, data: str, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, "w") as output:
        os.fchmod(output.fileno(), mode)
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def require_operator() -> None:
    if os.geteuid() != 0:
        raise ValueError("Apply requires the administrator-owned copy run with sudo.")
    script = Path(__file__).absolute()
    if script != PROTECTED_SCRIPT:
        raise ValueError(f"First install this script as {PROTECTED_SCRIPT}, root-owned mode 0700.")
    info = script.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("The protected script must be a root-owned regular file, mode 0700.")
    protected_directory(PROTECTED_SCRIPT.parent)


def preflight() -> tuple[int, int, Path]:
    require_operator()
    protected_directory(ROOT.parent)
    protected_directory(UNIT_DIR)
    account = pwd.getpwnam("demo")
    if account.pw_uid != 1001 or account.pw_gid != 1001:
        raise ValueError("This reviewed setup requires the dedicated demo account, UID/GID 1001.")
    for tool in [*(Path("/usr/bin") / name for name in TOOLS), Path("/usr/sbin/mkfs.ext4")]:
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise ValueError(f"Required installed tool is missing: {tool}")
    absent(ROOT)
    unit = UNIT_DIR / unit_name()
    absent(unit)
    absent(UNIT_DIR / "multi-user.target.wants" / unit.name)
    # Refuse alternative overrides or a previous partial installation too.
    for base in (Path("/run/systemd/system"), Path("/usr/lib/systemd/system"), UNIT_DIR):
        absent(base / (unit.name + ".d"))
        if base != UNIT_DIR:
            absent(base / unit.name)
    if shutil.disk_usage(ROOT.parent).free < SIZE + HEADROOM:
        raise ValueError("Need room for 128 GiB storage plus 32 GiB host headroom.")
    return account.pw_uid, account.pw_gid, unit


def verify_mount() -> None:
    mounted = run("/usr/bin/findmnt", "-n", "-o", "FSTYPE", "--mountpoint", str(MOUNT))
    if mounted != "ext4" or MOUNT.stat().st_dev == ROOT.stat().st_dev:
        raise ValueError("Expected separate ext4 filesystem was not mounted.")
    capacity = os.statvfs(MOUNT)
    if not SIZE * 0.9 < capacity.f_blocks * capacity.f_frsize <= SIZE:
        raise ValueError("Mounted filesystem has an unexpected capacity.")


def apply() -> dict:
    uid, gid, unit = preflight()
    os.umask(0o077)
    ROOT.mkdir(mode=0o755)
    os.chmod(ROOT, 0o755)
    # The unmounted target stays inaccessible to demo, preventing fallback writes.
    MOUNT.mkdir(mode=0o700)
    descriptor = os.open(IMAGE, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    run("/usr/bin/fallocate", "--length", str(SIZE), str(IMAGE))
    info = IMAGE.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_size != SIZE:
        raise ValueError("Refusing to format an unexpected target.")
    # Format only the new regular file, never a partition/device. Avoid discarding
    # its initial allocation; unrelated host writers still need their own budget.
    run(
        "/usr/sbin/mkfs.ext4",
        "-q",
        "-F",
        "-m",
        "0",
        "-L",
        "ML-DEMO",
        "-E",
        "nodiscard,lazy_itable_init=0,lazy_journal_init=0",
        str(IMAGE),
    )
    write_new(unit, unit_text(), 0o644)
    run("/usr/bin/systemctl", "daemon-reload")
    run("/usr/bin/systemctl", "start", unit.name)
    verify_mount()
    os.chmod(MOUNT, 0o711)
    for name in ("workspace", "tmp", "native-sockets", "logs"):
        target = MOUNT / name
        target.mkdir(mode=0o700)
        os.chown(target, uid, gid)
    run("/usr/bin/systemctl", "enable", unit.name)
    result = {
        "status": "empty bounded storage ready; application unchanged",
        "image_bytes": SIZE,
        "mount": str(MOUNT),
        "mount_unit": unit.name,
        "data_moved": False,
        "application_restarted": False,
    }
    write_new(ROOT / "setup.json", json.dumps(result, indent=2) + "\n", 0o644)
    os.sync()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "changes": False,
                    "allocate_gib": 128,
                    "new_image": str(IMAGE),
                    "new_mount": str(MOUNT),
                    "scope": (
                        "Create empty persistent storage and boot mount; no data migration, "
                        "service restart, packages, firewall or DNS changes."
                    ),
                    "apply": f"sudo /usr/bin/python3 -I {PROTECTED_SCRIPT} --apply",
                    "on_failure": (
                        "Stop and report the error. Partial state is retained; "
                        "no automatic deletion or reformatting."
                    ),
                },
                indent=2,
            )
        )
        return
    try:
        print(json.dumps(apply(), indent=2))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(
            1, f"Storage setup stopped: {error}. Keep partial files; do not rerun or delete them.\n"
        )


if __name__ == "__main__":
    main()
