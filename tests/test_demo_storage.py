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

"""Storage setup checks; no root access, real formatting or mounting in tests."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo_storage.py"
SPEC = importlib.util.spec_from_file_location("demo_storage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "new-storage"
    units = tmp_path / "units"
    units.mkdir()
    monkeypatch.setattr(storage, "ROOT", root)
    monkeypatch.setattr(storage, "IMAGE", root / "demo-data.ext4")
    monkeypatch.setattr(storage, "MOUNT", root / "data")
    monkeypatch.setattr(storage, "UNIT_DIR", units)
    monkeypatch.setattr(storage, "SIZE", 16 * 1024**2)
    monkeypatch.setattr(storage, "HEADROOM", 1024**2)
    monkeypatch.setattr(storage, "require_operator", lambda: None)
    monkeypatch.setattr(storage, "protected_directory", lambda path: None)
    monkeypatch.setattr(
        storage.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1001, pw_gid=1001)
    )
    monkeypatch.setattr(storage, "verify_mount", lambda: None)
    monkeypatch.setattr(storage.os, "chown", lambda *args: None)
    monkeypatch.setattr(storage.os, "sync", lambda: None)
    commands = []

    def run(*args):
        commands.append(args)
        if args[0].endswith("systemd-escape"):
            return "monailabel-disposable-storage-test.mount"
        if args[0].endswith("fallocate"):
            os.truncate(storage.IMAGE, storage.SIZE)
        return ""

    monkeypatch.setattr(storage, "run", run)
    original = os.umask(0o022)
    try:
        yield SimpleNamespace(root=root, units=units, commands=commands)
    finally:
        os.umask(original)


def fake_image_owner(monkeypatch):
    real_lstat = Path.lstat

    def lstat(path):
        info = real_lstat(path)
        if path == storage.IMAGE:
            return SimpleNamespace(st_mode=info.st_mode, st_uid=0, st_size=info.st_size)
        return info

    monkeypatch.setattr(Path, "lstat", lstat)


def test_default_plan_is_non_mutating(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr(storage, "apply", lambda: pytest.fail("plan must not apply"))
    storage.main()
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "plan" and result["changes"] is False
    assert result["allocate_gib"] == 128


def test_apply_requires_root(monkeypatch):
    monkeypatch.setattr(storage.os, "geteuid", lambda: 1000)
    with pytest.raises(ValueError, match="administrator-owned"):
        storage.require_operator()


def test_root_cannot_run_service_owned_script(monkeypatch):
    monkeypatch.setattr(storage.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="First install"):
        storage.require_operator()


@pytest.mark.parametrize("kind", ["directory", "file", "dangling-link"])
def test_existing_storage_is_never_reformatted(setup, kind):
    if kind == "directory":
        setup.root.mkdir()
    elif kind == "file":
        setup.root.write_text("retain existing data")
    else:
        setup.root.symlink_to(setup.root.parent / "absent-target")
    with pytest.raises(ValueError, match="Already exists"):
        storage.apply()
    assert not setup.commands


def test_existing_mount_unit_is_not_replaced(setup):
    unit = setup.units / storage.unit_name()
    unit.write_text("existing mount")
    with pytest.raises(ValueError, match="Already exists"):
        storage.apply()
    assert unit.read_text() == "existing mount"
    assert not setup.root.exists()


def test_low_disk_space_refused_before_creation(setup, monkeypatch):
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=storage.SIZE)
    )
    with pytest.raises(ValueError, match="headroom"):
        storage.apply()
    assert not setup.root.exists()


def test_wrong_demo_account_refused(setup, monkeypatch):
    monkeypatch.setattr(
        storage.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1000, pw_gid=1000)
    )
    with pytest.raises(ValueError, match="UID/GID"):
        storage.apply()
    assert not setup.root.exists()


def test_write_new_refuses_existing_or_symlink(tmp_path):
    target = tmp_path / "existing"
    target.write_text("retain")
    link = tmp_path / "link"
    link.symlink_to(target)
    for path in (target, link):
        with pytest.raises(FileExistsError):
            storage.write_new(path, "replacement", 0o600)
    assert target.read_text() == "retain"


def test_guarded_flow_formats_only_new_regular_image(setup, monkeypatch):
    fake_image_owner(monkeypatch)
    result = storage.apply()
    assert result["data_moved"] is False and result["application_restarted"] is False
    formats = [call for call in setup.commands if call[0].endswith("mkfs.ext4")]
    assert len(formats) == 1 and formats[0][-1] == str(storage.IMAGE)
    assert "nodiscard,lazy_itable_init=0,lazy_journal_init=0" in formats[0]
    assert storage.IMAGE.stat().st_size == storage.SIZE
    unit = setup.units / storage.unit_name()
    assert "Options=loop,nodev,nosuid" in unit.read_text()
    assert f"Where={storage.MOUNT}" in unit.read_text()
    assert not any("restart" in call or "stop" in call for call in setup.commands)
    for name in ("workspace", "tmp", "native-sockets", "logs"):
        directory = storage.MOUNT / name
        assert directory.is_dir() and not list(directory.iterdir())
        assert directory.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError, match="Already exists"):
        storage.apply()
    assert len([call for call in setup.commands if call[0].endswith("mkfs.ext4")]) == 1


def test_failed_mount_retains_state_and_never_enables_boot_unit(setup, monkeypatch):
    fake_image_owner(monkeypatch)

    def fail():
        raise ValueError("Expected separate ext4 filesystem was not mounted")

    monkeypatch.setattr(storage, "verify_mount", fail)
    with pytest.raises(ValueError, match="not mounted"):
        storage.apply()
    assert storage.IMAGE.exists()
    assert not any("enable" in call for call in setup.commands)
    assert not (storage.ROOT / "setup.json").exists()
