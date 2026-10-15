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

import hashlib
import socket
import ssl
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from monailabel.core.errors import DomainError
from monailabel.server.desktops.gateway import MAX_BODY, DesktopGateway
from monailabel.server.tls import ensure_local_certificate

TOKEN = "disposable-native-session-credential"


@pytest.fixture
def gateway():
    calls = []
    active = [True]

    def authorize(token):
        assert token == TOKEN
        if not active[0]:
            raise DomainError("expired", status=401)

    def backend(request):
        calls.append(request)
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b'{"ok":true}'),
            headers={"Content-Type": "application/json", "Set-Cookie": "never-forward"},
        )

    with tempfile.TemporaryDirectory(prefix="monai-gateway-") as directory:
        instance = DesktopGateway(
            Path(directory) / "api.sock",
            "http://127.0.0.1:8123",
            hashlib.sha256(TOKEN.encode()).hexdigest(),
            authorize,
        )
        instance.http.close()
        instance.http = httpx.Client(transport=httpx.MockTransport(backend))
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=str(instance.path)),
            base_url="http://desktop.invalid",
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=5,
        ) as client:
            try:
                yield instance, client, calls, active
            finally:
                instance.close()


def test_gateway_only_forwards_to_fixed_backend_and_filters_headers(gateway):
    _, client, calls, _ = gateway
    response = client.post(
        "/api/assets/fixture/review-mask?revision=2",
        content=b"disposable-mask",
        headers={
            "Host": "attacker.invalid",
            "Cookie": "monailabel_session=other-user",
            "Origin": "http://attacker.invalid",
            "X-Forwarded-For": "127.0.0.1",
            "Content-Type": "application/octet-stream",
            "X-MONAILABEL-REVIEW": "fixture-metadata",
        },
    )
    assert response.status_code == 200, response.text
    assert "set-cookie" not in response.headers
    request = calls[0]
    assert str(request.url) == "http://127.0.0.1:8123/api/assets/fixture/review-mask?revision=2"
    assert request.content == b"disposable-mask"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert request.headers["Host"] == "127.0.0.1:8123"
    assert request.headers["X-MONAILABEL-REVIEW"] == "fixture-metadata"
    for header in ("cookie", "origin", "x-forwarded-for"):
        assert header not in request.headers


@pytest.mark.parametrize(
    "path",
    [
        "/api/auth/setup",
        "/api/auth/login",
        "/api/auth/token",
        "/api/auth/users",
        "/api/demo-access/reset",
        "/api/desktops",
        "/docs",
        "/api/%2e%2e/api/auth/setup",
        "/api/%252e%252e/api/auth/setup",
    ],
)
def test_gateway_refuses_non_viewer_paths_without_contacting_backend(gateway, path):
    _, client, calls, _ = gateway
    assert client.get(path).status_code == 403
    assert not calls


def test_gateway_authentication_and_expiry_are_checked_per_request(gateway):
    _, client, calls, active = gateway
    assert client.get("/api/auth/me").status_code == 200
    assert len(calls) == 1
    assert (
        client.get(
            "/api/auth/me", headers={"Authorization": "Bearer another-visitor-token"}
        ).status_code
        == 401
    )
    active[0] = False
    assert client.get("/api/auth/me").status_code == 401
    assert len(calls) == 1


def raw_request(gateway, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(gateway.path))
        connection.sendall(request)
        response = b""
        while block := connection.recv(65536):
            response += block
        return response


@pytest.mark.parametrize(
    "request_line,headers,status",
    [
        ("CONNECT 127.0.0.1:11434 HTTP/1.1", "", 403),
        ("GET http://127.0.0.1:11434/api/tags HTTP/1.1", "", 403),
        ("GET /api/../api/auth/setup HTTP/1.1", "", 403),
        ("POST /api/assets/a/review-mask HTTP/1.1", "Content-Length: -1\r\n", 400),
        (
            "POST /api/assets/a/review-mask HTTP/1.1",
            "Content-Length: 1\r\nContent-Length: 1\r\n",
            400,
        ),
        ("POST /api/assets/a/review-mask HTTP/1.1", "Transfer-Encoding: chunked\r\n", 400),
        ("POST /api/assets/a/review-mask HTTP/1.1", f"Content-Length: {MAX_BODY + 1}\r\n", 413),
    ],
)
def test_gateway_rejects_tunneling_and_ambiguous_or_oversized_bodies(
    gateway, request_line, headers, status
):
    instance, _, calls, _ = gateway
    response = raw_request(
        instance,
        (
            f"{request_line}\r\nHost: fixture\r\nAuthorization: Bearer {TOKEN}\r\n{headers}\r\n"
        ).encode(),
    )
    assert response.startswith(f"HTTP/1.0 {status} ".encode()), response
    assert not calls


def test_gateway_does_not_follow_or_return_redirects(gateway):
    instance, client, _, _ = gateway
    calls = []

    def redirect(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "http://127.0.0.1:11434/api/tags"})

    instance.http.close()
    instance.http = httpx.Client(transport=httpx.MockTransport(redirect))
    response = client.get("/api/auth/me")
    assert response.status_code == 502 and "location" not in response.headers
    assert len(calls) == 1


def test_gateway_rate_limit_is_bounded_and_recovers(gateway, monkeypatch):
    instance, client, calls, _ = gateway
    tick = [100.0]
    monkeypatch.setattr("monailabel.server.desktops.gateway.monotonic", lambda: tick[0])
    monkeypatch.setattr("monailabel.server.desktops.gateway.REQUESTS_PER_MINUTE", 2)
    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/auth/me").status_code == 200
    response = client.get("/api/auth/me")
    assert response.status_code == 429 and response.headers["retry-after"] == "60"
    assert len(calls) == 2 and len(instance._requests) == 2
    tick[0] += 60
    assert client.get("/api/auth/me").status_code == 200


def test_gateway_active_connections_are_bounded(gateway):
    instance, _, _, _ = gateway
    acquired = [instance.server.slots.acquire(blocking=False) for _ in range(4)]
    assert all(acquired)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(instance.path))
            assert connection.recv(1) == b""
    finally:
        for _ in acquired:
            instance.server.slots.release()


@pytest.mark.parametrize("https", [False, True])
def test_real_backend_streaming_and_verified_tls(tmp_path, https):
    body = b"disposable-image-bytes" * 10000

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/assets/fixture/image"
            assert self.headers["Authorization"] == f"Bearer {TOKEN}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    certificate = ensure_local_certificate(tmp_path, "127.0.0.1")
    with ThreadingHTTPServer(("127.0.0.1", 0), Backend) as backend:
        if https:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certificate.certificate, certificate.key)
            backend.socket = context.wrap_socket(backend.socket, server_side=True)
        worker = threading.Thread(target=backend.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory(prefix="monai-gateway-") as directory:
                gateway = DesktopGateway(
                    Path(directory) / "api.sock",
                    f"{'https' if https else 'http'}://127.0.0.1:{backend.server_port}",
                    hashlib.sha256(TOKEN.encode()).hexdigest(),
                    lambda token: None,
                    certificate.authority.read_text() if https else None,
                )
                try:
                    with httpx.Client(
                        transport=httpx.HTTPTransport(uds=str(gateway.path)),
                        base_url="http://desktop.invalid",
                        headers={"Authorization": f"Bearer {TOKEN}"},
                    ) as client:
                        response = client.get("/api/assets/fixture/image")
                        assert response.status_code == 200
                        assert response.content == body
                finally:
                    gateway.close()
        finally:
            backend.shutdown()
            worker.join(timeout=2)
