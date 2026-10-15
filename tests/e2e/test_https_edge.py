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

"""Loopback TLS edge using a cached rootless Nginx image and synthetic data."""

import json
import os
import secrets
import socket
import ssl
import threading
import time
from contextlib import ExitStack

import httpx
import pytest
import uvicorn
from chat_fixture import ScriptedChat
from fastapi import Request, WebSocket
from https_fixture import https_edge

from monailabel.server.app import create_app

pytestmark = pytest.mark.desktop_e2e


def test_real_https_proxy_cookie_origin_forwarding_stream_and_websocket(tmp_path, monkeypatch):
    image = os.environ.get("MONAILABEL_TEST_NGINX_IMAGE")
    if not image:
        pytest.skip("Set MONAILABEL_TEST_NGINX_IMAGE to an already installed Nginx image.")
    from wsproto import ConnectionType, WSConnection, events
    from wsproto.events import AcceptConnection, BytesMessage, CloseConnection, TextMessage

    monkeypatch.setenv("MONAILABEL_ALLOWED_HOSTS", "localhost,127.0.0.1")
    monkeypatch.setenv("MONAILABEL_DEMO_SELF_REGISTER", "0")
    monkeypatch.setenv("MONAILABEL_DEMO_HTTP_LIMITS", "1")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "1")
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_PREVIEW", "1")
    app = create_app(tmp_path / "workspace", chat_provider=ScriptedChat())

    @app.get("/api/test-transport")
    def transport(request: Request):
        return {
            "scheme": request.url.scheme,
            "client": request.client.host,
            "forwarded": request.headers.get("forwarded"),
            "host": request.headers.get("host"),
        }

    @app.get("/api/test-stream")
    def stream():
        from fastapi.responses import StreamingResponse

        return StreamingResponse(iter([b"abcd" * 16384] * 4), media_type="application/octet-stream")

    @app.websocket("/test-socket")
    async def echo(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"scheme": websocket.url.scheme})
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        await websocket.send_bytes(message["bytes"])
        await websocket.close()

    backend_socket = socket.socket()
    backend_socket.bind(("127.0.0.1", 0))
    backend_port = backend_socket.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            access_log=False,
            log_level="error",
            proxy_headers=True,
            forwarded_allow_ips="127.0.0.1",
            ws="wsproto",
            ws_max_size=16 * 1024 * 1024,
            ws_per_message_deflate=False,
            limit_concurrency=64,
        )
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[backend_socket]), daemon=True)
    thread.start()
    cleanup = ExitStack()
    password = secrets.token_urlsafe(32)
    visitor_password = secrets.token_urlsafe(32)
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{backend_port}", trust_env=False) as local:
            assert (
                local.post(
                    "/api/auth/setup",
                    json={
                        "username": "test-owner",
                        "password": password,
                    },
                ).status_code
                == 201
            )
            for username in ("invited-one", "invited-two"):
                assert (
                    local.post(
                        "/api/auth/demo-users",
                        json={"username": username, "password": visitor_password, "minutes": 60},
                    ).status_code
                    == 201
                )
        edge = cleanup.enter_context(https_edge(tmp_path / "edge", backend_port))
        url, https_port, context = edge.url, edge.port, edge.context
        with httpx.Client(base_url=url, verify=context, trust_env=False, timeout=10) as browser:
            for _ in range(100):
                try:
                    assert browser.get("/api/auth/status").status_code == 200
                    break
                except httpx.ConnectError:
                    time.sleep(0.05)
            else:
                pytest.fail("Disposable HTTPS edge did not start")
            with pytest.raises(httpx.ConnectError):
                httpx.get(url + "/api/auth/status", trust_env=False)
            with (
                pytest.raises(ssl.SSLError),
                context.wrap_socket(
                    socket.create_connection(("127.0.0.1", https_port), timeout=5),
                    server_hostname="unknown.invalid",
                ),
            ):
                pass
            with pytest.raises(httpx.RemoteProtocolError):
                browser.get("/api/auth/status", headers={"Host": "unknown.invalid"})
            for path in (
                "/api/auth/setup",
                "/api/auth/register-demo",
                "/api/auth/users",
                "/api/auth/demo-users",
                "/api/auth/token",
                "/api/auth/%73etup",
                "/api/auth/setup/",
                "//api/auth/setup",
                "/api/demo-access/reset",
                "/api/demo-access/reset-preview",
                "/api/demo-access/audit-status",
                "/docs",
                "/openapi.json",
            ):
                assert browser.post(url + path).status_code == 404
            response = browser.get(
                "/api/test-transport",
                headers={
                    "X-Forwarded-For": "198.51.100.7",
                    "X-Forwarded-Proto": "http",
                    "Forwarded": "for=198.51.100.7;proto=http",
                },
            )
            assert response.json() == {
                "scheme": "https",
                "client": "127.0.0.1",
                "forwarded": None,
                "host": f"localhost:{https_port}",
            }
            login = browser.post(
                "/api/auth/login",
                json={
                    "username": "invited-one",
                    "password": visitor_password,
                },
                headers={"Origin": url},
            )
            assert login.status_code == 200
            assert login.json()["is_admin"] is False
            cookie = login.headers["set-cookie"].casefold()
            assert "secure" in cookie and "httponly" in cookie and "samesite=strict" in cookie
            assert browser.get("/api/projects").status_code == 200
            assert browser.get("/api/auth/session").json()["demo"] is True
            # Public visitor features must not be caught by the private admin block.
            assert browser.get("/api/demo-access/training").json()["enabled"] is True
            assert (
                browser.post(
                    "/api/demo-access/events",
                    json={"kind": "open", "target": "training"},
                ).status_code
                == 204
            )
            full = browser.post(
                "/api/auth/login",
                json={"username": "invited-two", "password": visitor_password},
            )
            assert full.status_code == 429 and full.json()["code"] == "demo_busy"
            assert full.headers.get("retry-after")
            denied = browser.post(
                "/api/projects",
                json={"name": "Cross-origin"},
                headers={"Origin": "https://attacker.invalid"},
            )
            assert denied.status_code == 403
            assert browser.get("/api/test-stream").content == b"abcd" * 65536
            assert browser.get("/api/auth/status").json()["demo_registration"] == {"enabled": False}
            # Reject declared excess length before buffering or upstream dispatch.
            with context.wrap_socket(
                socket.create_connection(("127.0.0.1", https_port)), server_hostname="localhost"
            ) as oversized:
                oversized.sendall(
                    (
                        f"POST /api/projects HTTP/1.1\r\nHost: localhost:{https_port}\r\n"
                        "Content-Length: 134217729\r\nConnection: close\r\n\r\n"
                    ).encode()
                )
                assert b" 413 " in oversized.recv(1024)
            for _ in range(10):
                limited = browser.post(
                    "/api/auth/login", json={"username": "missing", "password": "wrong"}
                )
                if limited.status_code == 429:
                    break
            assert limited.status_code == 429 and limited.headers.get("retry-after")
        with context.wrap_socket(
            socket.create_connection(("127.0.0.1", https_port), timeout=10),
            server_hostname="localhost",
        ) as display:
            websocket = WSConnection(ConnectionType.CLIENT)
            display.sendall(
                websocket.send(
                    events.Request(host=f"localhost:{https_port}", target="/test-socket")
                )
            )
            hello, echoed = "", bytearray()
            while not echoed:
                data = display.recv(65536)
                assert data, "WebSocket closed before the echo"
                websocket.receive_data(data)
                for event in websocket.events():
                    if isinstance(event, AcceptConnection):
                        display.sendall(
                            websocket.send(BytesMessage(data=b"disposable binary display frame"))
                        )
                    elif isinstance(event, TextMessage):
                        hello += event.data
                    elif isinstance(event, BytesMessage):
                        echoed.extend(event.data)
                    elif isinstance(event, CloseConnection):
                        display.sendall(websocket.send(event.response()))
            assert json.loads(hello) == {"scheme": "wss"}
            assert echoed == b"disposable binary display frame"
        # Send an oversized *fragmented* message. It must close at the transport
        # boundary, before the ASGI application receives a reassembled payload.
        with context.wrap_socket(
            socket.create_connection(("127.0.0.1", https_port), timeout=20),
            server_hostname="localhost",
        ) as display:
            websocket = WSConnection(ConnectionType.CLIENT)
            display.sendall(
                websocket.send(
                    events.Request(host=f"localhost:{https_port}", target="/test-socket")
                )
            )
            accepted = False
            while not accepted:
                incoming = display.recv(65536)
                assert incoming
                websocket.receive_data(incoming)
                accepted = any(isinstance(event, AcceptConnection) for event in websocket.events())
            for _ in range(17):
                display.sendall(
                    websocket.send(BytesMessage(data=b"x" * (1024 * 1024), message_finished=False))
                )
            closed = False
            while not closed:
                incoming = display.recv(65536)
                assert incoming, "Oversized message did not receive a close frame"
                websocket.receive_data(incoming)
                for event in websocket.events():
                    assert not isinstance(event, BytesMessage), "Oversized payload reached the app"
                    if isinstance(event, CloseConnection):
                        assert event.code == 1009
                        closed = True
        with httpx.Client(
            transport=httpx.HTTPTransport(local_address="127.0.0.2"), trust_env=False
        ) as untrusted:
            response = untrusted.get(
                f"http://127.0.0.1:{backend_port}/api/test-transport",
                headers={"X-Forwarded-For": "198.51.100.7", "X-Forwarded-Proto": "https"},
            )
            assert response.json()["scheme"] == "http"
            assert response.json()["client"] == "127.0.0.2"
    finally:
        try:
            cleanup.close()
        finally:
            server.should_exit = True
            thread.join(timeout=30)
            backend_socket.close()
            assert not thread.is_alive()
