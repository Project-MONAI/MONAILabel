import os
import shutil
from pathlib import Path

import pytest
from chat_fixture import ScriptedChat
from fastapi.testclient import TestClient

from monailabel.server.app import create_app
from monailabel.server.workspace import configure_workspace
from monailabel.viewers.manager import ViewerManager
from monailabel.viewers.ohif import OhifManager


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in (
        "MONAILABEL_DATA_DIR",
        "MONAILABEL_CACHE_DIR",
        "MONAILABEL_DATASETS_DIR",
        "MONAILABEL_MODELS_DIR",
        "MONAILABEL_TOOLS_DIR",
        "MONAILABEL_QUPATH_DATA",
        "HF_HOME",
        "TORCH_HOME",
    ):
        # Register both removal and the upcoming configure_workspace changes for restoration.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    return tmp_path / "workspace"


def test_server_paths_keep_data_and_caches_inside_workspace(isolated_paths):
    workspace = configure_workspace(None)
    assert workspace == isolated_paths
    cache = workspace / ".cache"
    for key, suffix in (
        ("MONAILABEL_DATASETS_DIR", "datasets"),
        ("MONAILABEL_MODELS_DIR", "models"),
        ("MONAILABEL_TOOLS_DIR", "tools"),
        ("HF_HOME", "huggingface"),
        ("TORCH_HOME", "torch"),
    ):
        assert Path(os.environ[key]) == cache / suffix
    assert ViewerManager().root == OhifManager().root == cache / "tools"
    assert Path(os.environ["MONAILABEL_QUPATH_DATA"]).is_relative_to(workspace)


def test_explicit_workspace_and_cache_overrides(isolated_paths, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DATA_DIR", str(isolated_paths / "from-env"))
    monkeypatch.setenv("MONAILABEL_CACHE_DIR", str(isolated_paths / "custom-cache"))
    monkeypatch.setenv("MONAILABEL_MODELS_DIR", str(isolated_paths / "custom-models"))
    workspace = configure_workspace(Path("chosen"))
    assert workspace == isolated_paths.parent / "chosen"
    assert os.environ["MONAILABEL_DATA_DIR"] == str(workspace)
    assert ViewerManager().root == isolated_paths / "custom-cache" / "tools"
    assert Path(os.environ["MONAILABEL_MODELS_DIR"]) == isolated_paths / "custom-models"


def test_clearing_stopped_workspace_returns_to_first_setup(isolated_paths):
    workspace = configure_workspace(None)
    with TestClient(create_app(chat_provider=ScriptedChat())) as client:
        assert (
            client.post(
                "/api/auth/setup", json={"username": "before", "password": "before-reset-password"}
            ).status_code
            == 201
        )
        assert client.post("/api/demo").status_code == 201
        assert client.get("/api/projects").json()
    cache = workspace / ".cache" / "datasets"
    cache.mkdir(parents=True)
    (cache / "download.zip").write_bytes(b"cached fixture")
    shutil.rmtree(workspace)
    with TestClient(create_app(chat_provider=ScriptedChat())) as client:
        assert (
            client.post(
                "/api/auth/setup", json={"username": "after", "password": "after-reset-password"}
            ).status_code
            == 201
        )
        assert client.get("/api/projects").json() == []
        assert not cache.exists()
