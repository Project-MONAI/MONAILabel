"""Desktop ownership, launch selection, lifecycle and authenticated streaming."""

import socketserver
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import Request, WebSocketDisconnect

from monailabel.core.errors import DomainError
from monailabel.core.models import Job, Role
from monailabel.server.auth import LoginSession
from monailabel.server.console_api import local_desktop
from monailabel.server.desktops.models import DesktopSession


@pytest.fixture
def runtime(http, tmp_path, monkeypatch):
    runtime = http.app.state.services.desktops.runtime
    runtime.dist = tmp_path / "novnc"
    runtime.dist.mkdir()
    (runtime.dist / "vnc.html").write_text('<html><script type="module">upstream()</script></html>')
    running = set()
    calls = []

    def start(identifier, viewer, configuration, progress):
        calls.append((identifier, viewer, configuration))
        running.add(identifier)

    monkeypatch.setattr(runtime, "start", start)
    monkeypatch.setattr(runtime, "running", lambda identifier: identifier in running)
    monkeypatch.setattr(runtime, "stop", lambda identifier: running.discard(identifier))
    runtime.calls = calls
    return runtime


def launch(client, asset, **kwargs):
    query = "&".join(f"{key}={value}" for key, value in kwargs.items())
    result = client.wait(client.post(f"/api/assets/{asset['id']}/viewer?{query}")["id"])
    return result["url"].rsplit("/", 1)[-1]


@pytest.mark.parametrize(
    "hostname,client_host,expected",
    [
        ("localhost", "127.0.0.1", True),
        ("127.0.0.2", "127.0.0.1", True),
        ("[::1]", "::1", True),
        ("viewer.example", "127.0.0.1", False),
        ("localhost", "192.168.1.10", False),
        ("192.168.1.3", "127.0.0.1", False),
    ],
)
def test_local_launch_requires_loopback_address_and_connection(hostname, client_host, expected):
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "path": "/",
            "headers": [(b"host", hostname.encode())],
            "client": (client_host, 1234),
        }
    )
    assert local_desktop(request) is expected


def test_remote_launch_reuses_only_same_account_asset_and_mode(client, http, seeded, runtime):
    _, assets = seeded
    asset = assets[0]
    identifier = launch(client, asset)
    assert launch(client, asset) == identifier
    assert len(runtime.calls) == 1
    configuration = runtime.calls[0][2]
    service = http.app.state.services
    assert configuration["shared_filesystem"] is False
    assert configuration["asset_id"] == asset["id"]
    assert service.auth.authenticate(configuration["token"]).username == "owner"
    response = http.get(f"/api/desktops/{identifier}")
    assert response.status_code == 200
    assert "token" not in response.text and "credential" not in response.text
    assert http.get(f"/desktop/{identifier}").status_code == 200
    client_html = http.get(f"/desktop/{identifier}/client/vnc.html")
    assert "/static/desktop-client.js" in client_html.text
    assert "upstream()" not in client_html.text
    assert client_html.headers["cache-control"] == "no-store"
    assert "ws://testserver/" in client_html.headers["content-security-policy"]
    assert launch(client, assets[1]) != identifier
    client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": 0,
            "mask": client.get(f"/api/assets/{asset['id']}/fixture")["mask"],
            "covered_labels": [0, 1, 2],
        },
    )
    assert launch(client, asset, mode="review") != identifier


def test_browser_override_on_localhost(client, http, seeded, runtime):
    http.headers["host"] = "localhost"
    assert launch(client, seeded[1][0], target="browser")
    assert len(runtime.calls) == 1


def test_other_accounts_and_logged_out_clients_cannot_view_end_or_stream(
    client, http, seeded, runtime
):
    identifier = launch(client, seeded[1][0])
    service = http.app.state.services
    other = service.auth.create_user("other", "a-different-password")
    service.auth.add_member(seeded[0]["project_id"], other.id, [Role.ANNOTATOR])
    http.cookies.clear()
    http.post("/api/auth/login", json={"username": "other", "password": "a-different-password"})
    for path in (
        f"/api/desktops/{identifier}",
        f"/desktop/{identifier}",
        f"/desktop/{identifier}/client/vnc.html",
    ):
        assert http.get(path).status_code == 403
    assert http.delete(f"/api/desktops/{identifier}").status_code == 403
    assert http.post(f"/api/desktops/{identifier}/close-tab").status_code == 403
    assert http.get("/api/desktops").json() == []
    with (
        pytest.raises(WebSocketDisconnect),
        http.websocket_connect(
            f"/desktop/{identifier}/socket", headers={"origin": "http://testserver"}
        ),
    ):
        pass
    http.cookies.clear()
    assert http.get(f"/desktop/{identifier}").status_code == 401


def test_end_revokes_native_token_and_preserves_submitted_data(client, http, seeded, runtime):
    asset = seeded[1][0]
    identifier = launch(client, asset)
    token = runtime.calls[0][2]["token"]
    assert http.delete(f"/api/desktops/{identifier}").status_code == 204
    assert http.delete(f"/api/desktops/{identifier}").status_code == 204
    assert not runtime.running(identifier)
    assert http.get(f"/api/desktops/{identifier}").status_code == 410
    with pytest.raises(DomainError):
        http.app.state.services.auth.authenticate(token)
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == asset["revision"]
    assert launch(client, asset) != identifier


def test_active_desktop_prevents_deleting_its_source(client, http, seeded, runtime):
    asset = seeded[1][0]
    identifier = launch(client, asset)
    response = http.delete(f"/api/assets/{asset['id']}")
    assert response.status_code == 409
    assert runtime.running(identifier)
    assert http.delete(f"/api/desktops/{identifier}").status_code == 204
    assert http.delete(f"/api/assets/{asset['id']}").status_code == 200


def test_native_exit_cleans_up_session_and_revokes_credential(client, http, seeded, runtime):
    asset = seeded[1][0]
    identifier = launch(client, asset)
    token = runtime.calls[0][2]["token"]
    runtime.stop(identifier)  # The native process has exited independently of the browser.
    assert http.get(f"/api/desktops/{identifier}").status_code == 410
    service = http.app.state.services
    assert service.store.get(DesktopSession, identifier).ended
    with pytest.raises(DomainError):
        service.auth.authenticate(token)
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == asset["revision"]
    assert http.get("/api/desktops").json() == []


def test_runtime_check_failure_keeps_session_and_credential(
    client, http, seeded, runtime, monkeypatch
):
    identifier = launch(client, seeded[1][0])
    token = runtime.calls[0][2]["token"]

    def unavailable(identifier):
        raise DomainError("Docker is temporarily unavailable.", status=503)

    monkeypatch.setattr(runtime, "running", unavailable)
    assert http.get(f"/api/desktops/{identifier}").status_code == 503
    service = http.app.state.services
    assert not service.store.get(DesktopSession, identifier).ended
    assert service.auth.authenticate(token)


@pytest.fixture
def tab_close(http, runtime, monkeypatch):
    monkeypatch.setattr("monailabel.server.desktops.sessions.CLOSE_DELAY_SECONDS", 0.05)
    stopped = threading.Event()
    stop = runtime.stop

    def tracked_stop(identifier):
        stop(identifier)
        stopped.set()

    monkeypatch.setattr(runtime, "stop", tracked_stop)
    return http.app.state.services.desktops, stopped


def test_closing_last_tab_ends_session_and_revokes_token(client, http, seeded, runtime, tab_close):
    desktops, stopped = tab_close
    asset = seeded[1][0]
    identifier = launch(client, asset)
    token = runtime.calls[0][2]["token"]
    user = http.app.state.services.auth.authenticate(token)
    desktops.connected(identifier, user)
    assert http.post(f"/api/desktops/{identifier}/close-tab").status_code == 204
    desktops.disconnected(identifier)
    assert stopped.wait(2)
    # Serialize with the stop callback before checking the final record.
    with desktops.lock:
        assert http.app.state.services.store.get(DesktopSession, identifier).ended
    with pytest.raises(DomainError):
        http.app.state.services.auth.authenticate(token)
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == asset["revision"]


@pytest.mark.parametrize("action", ["refresh", "another_tab", "network_loss", "server_restart"])
def test_tab_cleanup_preserves_connected_or_interrupted_drafts(
    client, http, seeded, runtime, tab_close, action
):
    desktops, stopped = tab_close
    identifier = launch(client, seeded[1][0])
    user = http.app.state.services.auth.authenticate(runtime.calls[0][2]["token"])
    desktops.connected(identifier, user)
    if action == "another_tab":
        desktops.connected(identifier, user)
    if action != "network_loss":
        assert http.post(f"/api/desktops/{identifier}/close-tab").status_code == 204
    desktops.disconnected(identifier)
    if action == "refresh":
        desktops.connected(identifier, user)
    elif action == "server_restart":
        desktops.close()
    assert not stopped.wait(0.15)
    assert runtime.running(identifier)


def test_failed_start_revokes_token(client, http, seeded, runtime, monkeypatch):
    def failed(*args):
        raise DomainError("The runtime failed.")

    monkeypatch.setattr(runtime, "start", failed)
    job = client.post(f"/api/assets/{seeded[1][0]['id']}/viewer")
    with pytest.raises(RuntimeError, match="runtime failed"):
        client.wait(job["id"])
    service = http.app.state.services
    desktop = service.store.list(DesktopSession)[0]
    assert desktop.ended
    assert service.store.get(LoginSession, desktop.credential_id).expires_at <= datetime.now(UTC)


def test_slow_launch_preserves_existing_connections_and_pending_session(
    client, http, seeded, runtime, monkeypatch
):
    existing = launch(client, seeded[1][0])
    service = http.app.state.services
    desktops = service.desktops
    user = service.auth.authenticate(runtime.calls[0][2]["token"])
    entered, release = threading.Event(), threading.Event()
    start = runtime.start

    def slow_start(*args):
        entered.set()
        assert release.wait(5)
        start(*args)

    monkeypatch.setattr(runtime, "start", slow_start)
    job = client.post(f"/api/assets/{seeded[1][1]['id']}/viewer")
    with ThreadPoolExecutor(max_workers=2) as requests:
        try:
            assert entered.wait(2)
            pending = next(s for s in service.store.list(DesktopSession) if s.id != existing)
            requests.submit(desktops.connected, existing, user).result(timeout=2)
            response = requests.submit(http.get, f"/api/desktops/{pending.id}").result(timeout=2)
            assert response.status_code == 409
            assert not service.store.get(DesktopSession, pending.id).ended
            assert runtime.running(existing)
        finally:
            release.set()
    result = client.wait(job["id"])
    assert result["url"] == f"/desktop/{pending.id}"
    assert http.get(f"/api/desktops/{pending.id}").status_code == 200
    desktops.disconnected(existing)


@pytest.mark.parametrize("action", ["end", "cancel", "disable"])
def test_launch_interrupted_during_setup_leaves_no_runtime_or_credential(
    client, http, seeded, runtime, monkeypatch, action
):
    service = http.app.state.services
    start, stop = runtime.start, runtime.stop
    stopped = threading.Event()

    def interrupted_start(identifier, viewer, configuration, progress):
        user = service.auth.authenticate(configuration["token"])
        if action == "end":
            service.desktops.end(identifier, user)
        elif action == "cancel":
            job = next(j for j in service.store.list(Job) if j.kind == "viewer")
            service.jobs.cancel(job.id)
        else:
            with service.store.transaction() as session:
                session.update(user.model_copy(update={"active": False}))
        start(identifier, viewer, configuration, progress)

    def tracked_stop(identifier):
        stop(identifier)
        stopped.set()

    monkeypatch.setattr(runtime, "start", interrupted_start)
    monkeypatch.setattr(runtime, "stop", tracked_stop)
    client.post(f"/api/assets/{seeded[1][0]['id']}/viewer")
    assert stopped.wait(2)
    # The explicit end may happen before startup resumes; wait for the worker cleanup.
    with service.desktops.launch_lock:
        desktop = service.store.list(DesktopSession)[0]
        assert desktop.ended
        assert not runtime.running(desktop.id)
        assert desktop.id not in service.desktops.starting
    with pytest.raises(DomainError):
        service.auth.authenticate(runtime.calls[0][2]["token"])


def test_reconnect_renews_bridge_token_without_restarting(client, http, seeded, runtime):
    identifier = launch(client, seeded[1][0])
    service = http.app.state.services
    desktop = service.store.get(DesktopSession, identifier)
    with service.store.transaction() as session:
        credential = session.get(LoginSession, desktop.credential_id)
        session.update(credential.model_copy(update={"expires_at": datetime.now(UTC)}))
    assert launch(client, seeded[1][0]) == identifier
    assert service.auth.authenticate(runtime.calls[0][2]["token"])
    assert len(runtime.calls) == 1


def test_session_limit_preserves_existing_drafts(client, http, seeded, runtime, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DESKTOP_LIMIT", "1")
    identifier = launch(client, seeded[1][0])
    job = client.post(f"/api/assets/{seeded[1][1]['id']}/viewer")
    with pytest.raises(RuntimeError, match="slots are in use"):
        client.wait(job["id"])
    assert runtime.running(identifier)
    assert launch(client, seeded[1][0]) == identifier


def test_exited_desktops_release_slots_without_replacing_running_drafts(
    client, seeded, runtime, monkeypatch
):
    monkeypatch.setenv("MONAILABEL_DESKTOP_LIMIT", "1")
    identifier = launch(client, seeded[1][0])
    runtime.stop(identifier)
    assert launch(client, seeded[1][1]) != identifier


def test_secure_display_policy(http, client, seeded, runtime):
    identifier = launch(client, seeded[1][0])
    http.base_url = "https://testserver"
    policy = http.get(f"/desktop/{identifier}").headers["content-security-policy"]
    assert "wss://testserver/" in policy
    assert "ws://testserver/" not in policy


def test_websocket_streams_bytes_and_rechecks_access(client, http, seeded, runtime, monkeypatch):
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.sendall(b"RFB 003.008\n")
            while data := self.request.recv(1024):
                self.request.sendall(data)

    identifier = launch(client, seeded[1][0])
    with tempfile.TemporaryDirectory(prefix="monai-rfb-") as directory:
        socket = Path(directory) / "rfb.sock"
        with socketserver.ThreadingUnixStreamServer(str(socket), Echo) as server:
            server.daemon_threads = True
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            monkeypatch.setattr(runtime, "socket", lambda _: socket)
            try:
                for origin in ("https://other.example", "null", "https://testserver", ""):
                    with (
                        pytest.raises(WebSocketDisconnect),
                        http.websocket_connect(
                            f"/desktop/{identifier}/socket", headers={"origin": origin}
                        ),
                    ):
                        pass
                with http.websocket_connect(
                    f"/desktop/{identifier}/socket", headers={"origin": "http://testserver"}
                ) as websocket:
                    assert websocket.receive_bytes() == b"RFB 003.008\n"
                    websocket.send_bytes(b"keyboard and pointer input")
                    assert websocket.receive_bytes() == b"keyboard and pointer input"
                    http.post("/api/auth/logout")
                    with pytest.raises(WebSocketDisconnect):
                        websocket.receive_bytes()
            finally:
                server.shutdown()
                worker.join()
