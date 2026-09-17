import pytest
from chat_fixture import ScriptedChat
from fastapi.testclient import TestClient

from monailabel.client.client import Client
from monailabel.server.app import create_app


@pytest.fixture(autouse=True)
def isolated_presets(monkeypatch, tmp_path):
    monkeypatch.setenv("MONAILABEL_PRELOAD_MODELS", "0")
    monkeypatch.setenv("MONAILABEL_DATASETS_DIR", str(tmp_path / "download-cache"))


@pytest.fixture
def http(tmp_path):
    with TestClient(create_app(tmp_path, chat_provider=ScriptedChat())) as http:
        response = http.post(
            "/api/auth/setup", json={"username": "owner", "password": "test-password-1234"}
        )
        assert response.status_code == 201, response.text
        yield http


@pytest.fixture
def client(http):
    return Client(http=http)


@pytest.fixture
def seeded(client):
    setup = client.post("/api/demo")
    assets = client.get(f"/api/projects/{setup['project_id']}/assets")
    return setup, assets
