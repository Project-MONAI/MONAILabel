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

"""A bounded, session-authenticated Unix-socket path to one fixed backend.

Native desktops have no IP network. This gateway is not a forward proxy: clients
cannot select an upstream, forward cookies, mint credentials, or issue CONNECT.
The normal backend still performs resource authorization on every request.
"""

import hashlib
import hmac
import math
import os
import re
import socket
import socketserver
import ssl
import threading
from collections import deque
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

import httpx

from monailabel.core.errors import DomainError

MAX_BODY = 128 * 1024 * 1024
MAX_CONNECTIONS = 4
REQUESTS_PER_MINUTE = 300


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = MAX_CONNECTIONS

    def __init__(self, gateway: "DesktopGateway") -> None:
        self.gateway = gateway
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(str(gateway.path), _Handler)

    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: str
    ) -> None:
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: str
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: str
    ) -> None:
        # An untrusted native script can close a connection at any point. Never
        # log its request headers, bearer credential, or upstream response body.
        pass


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    timeout = 10
    protocol_version = "HTTP/1.0"  # One request per connection; no pipelining.

    def log_message(self, format: str, *args: object) -> None:
        pass

    def reply(self, status: int, detail: str, *, retry: int | None = None) -> None:
        body = detail.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if retry is not None:
            self.send_header("Retry-After", str(retry))
        self.end_headers()
        self.wfile.write(body)

    def proxy(self) -> None:
        gateway = self.server.gateway
        if gateway.closed:
            self.reply(410, "This desktop connection has ended.")
            return
        if retry := gateway.rate_limit():
            self.reply(429, "Desktop request limit reached.", retry=retry)
            return
        path = urlsplit(self.path)
        if (
            self.command not in {"GET", "POST", "PUT", "PATCH", "DELETE"}
            or path.scheme
            or path.netloc
            or path.fragment
            or not re.fullmatch(r"/api/[A-Za-z0-9_./-]+", path.path)
            or "//" in path.path
            or any(part in {".", ".."} for part in path.path.split("/"))
            or (
                path.path.startswith("/api/auth/")
                and (
                    self.command != "GET" or path.path not in {"/api/auth/me", "/api/auth/session"}
                )
            )
            or path.path.startswith(("/api/demo-access/", "/api/desktops"))
        ):
            self.reply(403, "This endpoint is unavailable from a native desktop.")
            return
        credentials = self.headers.get_all("Authorization", [])
        if len(credentials) != 1 or not credentials[0].startswith("Bearer "):
            self.reply(401, "A desktop credential is required.")
            return
        token = credentials[0].removeprefix("Bearer ")
        if not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), gateway.credential):
            self.reply(401, "Invalid desktop credential.")
            return
        try:
            gateway.authorize(token)
        except DomainError:
            self.reply(401, "Desktop access has expired or been revoked.")
            return
        if sum(len(key) + len(value) for key, value in self.headers.items()) > 65536:
            self.reply(431, "Request headers are too large.")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if (
            self.headers.get("Transfer-Encoding") is not None
            or len(lengths) > 1
            or (lengths and not re.fullmatch(r"[0-9]{1,10}", lengths[0]))
        ):
            self.reply(400, "Send a single, bounded Content-Length.")
            return
        length = int(lengths[0]) if lengths else 0
        if length > MAX_BODY:
            self.reply(413, "Desktop request body exceeds 128 MiB.")
            return
        deadline = monotonic() + 180

        def body() -> Iterator[bytes]:
            remaining = length
            while remaining:
                if gateway.closed or monotonic() >= deadline:
                    raise TimeoutError("Desktop request ended")
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    raise ConnectionError("Incomplete desktop body")
                remaining -= len(chunk)
                yield chunk

        headers = {"Authorization": credentials[0], "Content-Length": str(length)}
        # Do not relay Host, Cookie, Origin, forwarding headers or hop-by-hop headers.
        for name in ("Content-Type", "X-MONAILABEL-REVIEW"):
            if value := self.headers.get(name):
                headers[name] = value
        started = False
        try:
            with gateway.http.stream(
                self.command, gateway.backend + self.path, headers=headers, content=body()
            ) as response:
                if 300 <= response.status_code < 400:
                    self.reply(502, "The desktop backend must not redirect requests.")
                    return
                self.send_response(response.status_code)
                for name in ("Content-Type", "Content-Length", "Content-Encoding", "Retry-After"):
                    if value := response.headers.get(name):
                        self.send_header(name, value)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                for chunk in response.iter_raw(chunk_size=65536):
                    if gateway.closed or monotonic() >= deadline:
                        break
                    self.wfile.write(chunk)
        except (httpx.HTTPError, OSError):
            if not started:
                self.reply(502, "The desktop backend is unavailable.")

    do_GET = proxy
    do_POST = proxy
    do_PUT = proxy
    do_PATCH = proxy
    do_DELETE = proxy
    do_CONNECT = proxy


class DesktopGateway:
    def __init__(
        self,
        path: Path,
        backend: str,
        credential: str,
        authorize: Callable[[str], None],
        ca_certificate: str | None = None,
    ):
        parsed = urlsplit(backend)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise DomainError("Configure a fixed HTTP(S) desktop backend origin.", status=503)
        self.path, self.backend = path, backend.rstrip("/")
        self.credential, self.authorize = credential, authorize
        self.closed = False
        self._lock = threading.Lock()
        self._requests: deque[float] = deque()
        context = ssl.create_default_context(
            cadata=ca_certificate if parsed.scheme == "https" else None
        )
        self.http = httpx.Client(
            verify=context,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(180, connect=5),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
        )
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(path.parent, 0o700)
        # A workspace lock excludes another server; old sockets cannot survive a restart.
        path.unlink(missing_ok=True)
        self.server = _Server(self)
        os.chmod(path, 0o600)
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.1),
            name="desktop-gateway",
            daemon=True,
        )
        self.thread.start()

    def rate_limit(self) -> int:
        with self._lock:
            now = monotonic()
            while self._requests and self._requests[0] <= now - 60:
                self._requests.popleft()
            if len(self._requests) >= REQUESTS_PER_MINUTE:
                return max(1, math.ceil(self._requests[0] + 60 - now))
            self._requests.append(now)
            return 0

    def close(self) -> None:
        self.closed = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.http.close()
        self.path.unlink(missing_ok=True)
