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

"""Startup guards use disposable files; no mounting or application mutation."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo_storage_run.py"
SPEC = importlib.util.spec_from_file_location("demo_storage_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def bounded(tmp_path, monkeypatch):
    root = tmp_path / "storage"
    root.mkdir(mode=0o711)
    for name in ("workspace", "tmp", "native-sockets", "logs"):
        (root / name).mkdir(mode=0o700)
    workspace = root / "workspace/live"
    workspace.mkdir(mode=0o700)
    for name in ("monailabel.sqlite3", "secrets.key"):
        (workspace / name).touch(mode=0o600)
    # Model a short reviewed alias without creating anything outside tmp_path.
    alias = Path("/short-demo-sockets")
    real_lstat, real_resolve = Path.lstat, Path.resolve

    def lstat(path):
        value = real_lstat(path)
        if path == root:
            return SimpleNamespace(st_uid=0, st_mode=value.st_mode, st_dev=value.st_dev)
        return value

    def resolve(path, *args, **kwargs):
        return root / "native-sockets" if path == alias else real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "is_mount", lambda path: path == root)
    monkeypatch.setattr(
        runner.os,
        "statvfs",
        lambda path: SimpleNamespace(f_blocks=127 * 1024**3 // 4096, f_frsize=4096),
    )
    return SimpleNamespace(root=root, workspace=workspace, alias=alias)


def settings(bounded):
    return runner.environment(bounded.root, bounded.workspace, bounded.alias)


def test_overrides_storage_but_preserves_existing_application_configuration(bounded, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/old-unbounded-tmp")
    monkeypatch.setenv("MONAILABEL_DATA_DIR", "/old-workspace")
    monkeypatch.setenv("MONAILABEL_MODELS_DIR", "/existing-read-only-model-cache")
    result = settings(bounded)
    assert result["MONAILABEL_DATA_DIR"] == str(bounded.workspace)
    assert result["TMPDIR"] == result["TEMP"] == result["TMP"] == str(bounded.root / "tmp")
    assert result["MONAILABEL_DESKTOP_SOCKET_DIR"] == str(bounded.alias)
    assert result["MONAILABEL_MODELS_DIR"] == "/existing-read-only-model-cache"
    assert os.environ["MONAILABEL_DATA_DIR"] == "/old-workspace"


def test_missing_mount_fails_closed(bounded, monkeypatch):
    monkeypatch.setattr(Path, "is_mount", lambda path: False)
    with pytest.raises(ValueError, match="not mounted"):
        settings(bounded)


def test_root_application_refused(bounded, monkeypatch):
    monkeypatch.setattr(runner.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="non-root"):
        settings(bounded)


def test_unexpected_capacity_refused(bounded, monkeypatch):
    monkeypatch.setattr(
        runner.os, "statvfs", lambda path: SimpleNamespace(f_blocks=1024**4 // 4096, f_frsize=4096)
    )
    with pytest.raises(ValueError, match="capacity"):
        settings(bounded)


@pytest.mark.parametrize("name", ["workspace", "tmp", "native-sockets", "logs"])
def test_nonprivate_storage_refused(bounded, name):
    (bounded.root / name).chmod(0o755)
    with pytest.raises(ValueError, match="private directory"):
        settings(bounded)


@pytest.mark.parametrize("name", ["monailabel.sqlite3", "secrets.key"])
def test_missing_database_or_key_refuses_new_empty_demo(bounded, name):
    (bounded.workspace / name).unlink()
    with pytest.raises(FileNotFoundError):
        settings(bounded)


def test_workspace_outside_mount_refused(bounded, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="inside"):
        runner.environment(bounded.root, outside, bounded.alias)


def test_linked_workspace_refused(bounded):
    link = bounded.root / "workspace/link"
    link.symlink_to(bounded.workspace)
    with pytest.raises(ValueError, match="private directory"):
        runner.environment(bounded.root, link, bounded.alias)


def test_wrong_socket_alias_refused(bounded):
    with pytest.raises(ValueError, match="Socket alias"):
        runner.environment(bounded.root, bounded.workspace, Path("/tmp"))


def test_direct_long_socket_path_refused(bounded):
    with pytest.raises(ValueError, match="shorter socket alias"):
        runner.environment(bounded.root, bounded.workspace, bounded.root / "native-sockets")
