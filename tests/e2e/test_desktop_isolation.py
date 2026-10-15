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

"""Network-boundary proof with a cached Python-capable desktop image; no downloads.

Run with --desktop-e2e and MONAILABEL_TEST_DESKTOP_IMAGE set to an installed image.
This exercises the gateway/relay boundary, not native viewer editing workflows.
"""

import hashlib
import json
import os
import secrets
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from monailabel.core.errors import DomainError
from monailabel.server.desktops.gateway import DesktopGateway
from monailabel.viewers.browser_desktop import RESOURCES

pytestmark = pytest.mark.desktop_e2e


def test_real_networkless_desktop_gateway_and_restart(tmp_path):
    image = os.environ.get("MONAILABEL_TEST_DESKTOP_IMAGE")
    if not image:
        pytest.skip("Set MONAILABEL_TEST_DESKTOP_IMAGE to an already installed desktop image.")
    token = secrets.token_urlsafe(32)
    calls = []
    revoked = threading.Event()

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, dict(self.headers.items())))
            body = b"disposable-isolation-proof"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    def authorize(credential):
        assert credential == token
        if revoked.is_set():
            raise DomainError("Disposable credential revoked", status=401)

    def docker(*args, input=None):
        return subprocess.run(
            ["docker", *args], input=input, text=True, capture_output=True, timeout=30, check=True
        ).stdout

    # Refuse to pull or build anything during the boundary test.
    docker("image", "inspect", image)
    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    backend.daemon_threads = True
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    port = backend.server_address[1]
    gateway = None
    name = "monailabel-isolation-test-" + secrets.token_hex(6)
    created = False
    # pytest's descriptive directory can exceed the Unix socket's path limit.
    socket_root = tmp_path.parent / ("gateway-" + secrets.token_hex(4))
    socket_path = socket_root / "api.sock"

    def open_gateway():
        return DesktopGateway(
            socket_path,
            f"http://127.0.0.1:{port}",
            hashlib.sha256(token.encode()).hexdigest(),
            authorize,
        )

    def probe(expected):
        # Credentials travel via stdin, never container arguments or result logs.
        source = f"""
import http.client, json, os, socket, time
token = {json.dumps(token)}
expected = {expected}
for attempt in range(100):
    try:
        connection = http.client.HTTPConnection('127.0.0.1', 8765, timeout=5)
        connection.request('GET', '/api/projects', headers={{
            'Authorization': 'Bearer ' + token,
            'Cookie': 'must-not-forward=1', 'X-Forwarded-For': '198.51.100.1'}})
        response = connection.getresponse()
        assert response.status == expected, response.status
        body = response.read()
        if expected == 200:
            assert body == b'disposable-isolation-proof'
        connection.close()
        break
    except ConnectionRefusedError:
        time.sleep(0.05)
else:
    raise AssertionError('Relay did not start')
assert os.listdir('/sys/class/net') == ['lo']
for target in [('127.0.0.1', {port}), ('127.0.0.1', 8013), ('192.0.2.1', 443)]:
    try:
        with socket.create_connection(target, timeout=1):
            raise AssertionError('Unexpected direct network connection')
    except OSError:
        pass
try:
    os.unlink('/gateway/api.sock')
except OSError:
    pass
else:
    raise AssertionError('Gateway mount is writable')
for path, credential, status in [
    ('/api/auth/tokens', token, 403),
    ('/api/projects', 'another-desktop-token', 401),
]:
    connection = http.client.HTTPConnection('127.0.0.1', 8765, timeout=5)
    connection.request('GET', path, headers={{'Authorization': 'Bearer ' + credential}})
    response = connection.getresponse()
    assert response.status == status, response.status
    response.read()
    connection.close()
print('boundary checks passed')
"""
        assert "boundary checks passed" in docker("exec", "-i", name, "python3", "-", input=source)

    try:
        gateway = open_gateway()
        docker(
            "run",
            "-d",
            "--pull=never",
            "--name",
            name,
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--pids-limit",
            "32",
            "--memory",
            "256m",
            "--cpus",
            "0.5",
            "--tmpfs",
            "/tmp:rw,nosuid,noexec,size=16m",
            "--log-opt",
            "max-size=1m",
            "--log-opt",
            "max-file=1",
            "--volume",
            f"{socket_root}:/gateway:ro",
            "--volume",
            f"{RESOURCES / 'desktop/relay.py'}:/opt/isolation-relay.py:ro",
            "--entrypoint",
            "python3",
            image,
            "/opt/isolation-relay.py",
        )
        created = True
        probe(200)
        assert len(calls) == 1
        assert not any(key.casefold() in {"cookie", "x-forwarded-for"} for key in calls[0][1])
        gateway.close()
        gateway = open_gateway()
        probe(200)
        assert len(calls) == 2
        revoked.set()
        probe(401)
        assert len(calls) == 2  # Revocation was checked before reaching the backend.
    finally:
        if created:
            docker("rm", "-f", name)
        if gateway:
            gateway.close()
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)
        if socket_root.exists():
            socket_root.rmdir()
