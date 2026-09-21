import pytest
from chat_fixture import ScriptedChat
from fastapi.testclient import TestClient

from monailabel.client.client import Client
from monailabel.server.app import create_app


def pytest_addoption(parser):
    parser.addoption(
        "--browser-e2e", action="store_true", help="Run disposable workspace browser tests."
    )
    parser.addoption(
        "--desktop-e2e", action="store_true", help="Run disposable browser desktop tests."
    )
    parser.addoption(
        "--video-e2e", action="store_true", help="Run disposable CVAT browser integration tests."
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--browser-e2e"):
        skip = pytest.mark.skip(reason="Use --browser-e2e with the e2e dependency group.")
        for item in items:
            if "browser_e2e" in item.keywords:
                item.add_marker(skip)
    if not config.getoption("--desktop-e2e"):
        skip = pytest.mark.skip(reason="Use --desktop-e2e with the e2e dependency group.")
        for item in items:
            if "desktop_e2e" in item.keywords:
                item.add_marker(skip)
    if not config.getoption("--video-e2e"):
        skip = pytest.mark.skip(reason="Use --video-e2e with the e2e dependency group.")
        for item in items:
            if "video_e2e" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_presets(monkeypatch, tmp_path):
    monkeypatch.setenv("MONAILABEL_PRELOAD_MODELS", "0")
    monkeypatch.setenv("MONAILABEL_DATASETS_DIR", str(tmp_path / "download-cache"))


@pytest.fixture
def http(tmp_path):
    with TestClient(
        create_app(tmp_path, chat_provider=ScriptedChat()), client=("127.0.0.1", 50000)
    ) as http:
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
