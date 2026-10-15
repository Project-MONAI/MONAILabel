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

"""Backup refusal and restore checks use only disposable workspaces."""

import fcntl
import importlib.util
import json
import os
import socket
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo_backup.py"
SPEC = importlib.util.spec_from_file_location("demo_backup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


@pytest.fixture
def sample(tmp_path):
    workspace = tmp_path / "workspace"
    destination = tmp_path / "backups"
    workspace.mkdir(mode=0o700)
    destination.mkdir(mode=0o700)
    (workspace / "server.lock").touch()
    (workspace / "secrets.key").write_bytes(b"synthetic-key-only")
    with sqlite3.connect(workspace / "monailabel.sqlite3") as database:
        database.execute("CREATE TABLE annotations (id TEXT, revision INTEGER)")
        database.execute("INSERT INTO annotations VALUES ('synthetic-annotation', 3)")
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts/mask").write_bytes(b"synthetic immutable annotation")
    (workspace / "desktop/drafts").mkdir(parents=True)
    (workspace / "desktop/drafts/unsent.mask").write_bytes(b"saved viewer draft")
    (workspace / ".cache").mkdir()
    (workspace / ".cache/reproducible-download").write_bytes(b"download")
    return SimpleNamespace(workspace=workspace, destination=destination, root=tmp_path)


def create(sample):
    result = backup.create_backup(
        sample.workspace, sample.destination, apply=True, writers_stopped=True
    )
    return Path(result["backup"])


def test_plan_has_no_writes(sample):
    before = {path: path.read_bytes() for path in sample.workspace.rglob("*") if path.is_file()}
    result = backup.create_backup(
        sample.workspace, sample.destination, apply=False, writers_stopped=False
    )
    assert result["mode"] == "plan" and result["changes"] is False
    assert not list(sample.destination.iterdir())
    assert before == {path: path.read_bytes() for path in before}


def test_active_server_refused_before_creating_backup(sample):
    with (sample.workspace / "server.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            create(sample)
    assert not list(sample.destination.iterdir())


def test_requires_explicit_writer_confirmation(sample):
    with pytest.raises(ValueError, match="Confirm"):
        backup.create_backup(
            sample.workspace, sample.destination, apply=True, writers_stopped=False
        )
    assert not list(sample.destination.iterdir())


def test_restore_retains_database_drafts_key_and_immutable_files(sample):
    snapshot = create(sample)
    assert snapshot.stat().st_mode & 0o777 == 0o700
    manifest = backup.verify_backup(snapshot)
    assert "server.lock" not in manifest["files"]
    assert not any(name.startswith(".cache/") for name in manifest["files"])
    target = sample.destination / "restore-test"
    backup.restore_copy(snapshot, target)
    with sqlite3.connect(f"{target.as_uri()}/monailabel.sqlite3?mode=ro", uri=True) as database:
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert database.execute("SELECT revision FROM annotations").fetchone() == (3,)
    assert (target / "secrets.key").read_bytes() == b"synthetic-key-only"
    assert (target / "desktop/drafts/unsent.mask").read_bytes() == b"saved viewer draft"
    assert (target / "artifacts/mask").read_bytes() == b"synthetic immutable annotation"
    assert all(path.stat().st_mode & 0o077 == 0 for path in target.rglob("*"))
    assert (sample.workspace / "server.lock").exists()


def test_restore_never_overwrites_existing_directory(sample):
    snapshot = create(sample)
    target = sample.destination / "existing"
    target.mkdir()
    (target / "keep").write_text("newer annotation")
    with pytest.raises(ValueError, match="already exists"):
        backup.restore_copy(snapshot, target)
    assert (target / "keep").read_text() == "newer annotation"


def test_tampering_refused_before_restore(sample):
    snapshot = create(sample)
    (snapshot / "workspace/secrets.key").write_bytes(b"different-key")
    target = sample.destination / "restore-test"
    with pytest.raises(ValueError, match="checksum"):
        backup.restore_copy(snapshot, target)
    assert not target.exists()


@pytest.mark.parametrize("entry", ["linked-file", "linked-directory"])
def test_linked_data_requires_separate_plan(sample, entry):
    (sample.workspace / entry).symlink_to(sample.root if entry.endswith("directory") else SCRIPT)
    with pytest.raises(ValueError, match="Linked"):
        create(sample)
    assert not list(sample.destination.iterdir())


def test_sockets_refused_before_backup(sample, monkeypatch):
    monkeypatch.chdir(sample.workspace)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind("gateway.sock")
        with pytest.raises(ValueError, match="socket"):
            create(sample)
    assert not list(sample.destination.iterdir())


def test_destination_inside_workspace_refused(sample):
    nested = sample.workspace / "backups"
    nested.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="outside"):
        backup.create_backup(sample.workspace, nested, apply=False, writers_stopped=False)


def test_world_readable_backup_directory_refused(sample):
    sample.destination.chmod(0o755)
    with pytest.raises(ValueError, match="accessible only"):
        create(sample)


def test_insufficient_disk_space_refused(sample, monkeypatch):
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="headroom"):
        create(sample)
    assert not list(sample.destination.iterdir())


def test_failed_copy_keeps_incomplete_directory_without_publishing(sample, monkeypatch):
    def fail(*args):
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(backup, "copy_checked", fail)
    with pytest.raises(OSError, match="synthetic"):
        create(sample)
    assert len(list(sample.destination.glob(".incomplete-*"))) == 1
    assert not list(sample.destination.glob("backup-*"))


def test_manifest_cannot_restore_outside_target(sample):
    snapshot = create(sample)
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = {"sha256": "none", "bytes": 0}
    path.write_text(json.dumps(manifest))
    target = sample.destination / "restore-test"
    with pytest.raises(ValueError):
        backup.restore_copy(snapshot, target)
    assert not target.exists()


def test_multiple_backups_do_not_replace_older_versions(sample):
    first = create(sample)
    (sample.workspace / "desktop/drafts/unsent.mask").write_bytes(b"newer draft")
    second = create(sample)
    assert first != second
    assert (first / "workspace/desktop/drafts/unsent.mask").read_bytes() == b"saved viewer draft"
    assert (second / "workspace/desktop/drafts/unsent.mask").read_bytes() == b"newer draft"
    assert os.stat(first).st_ino != os.stat(second).st_ino
