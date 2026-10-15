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

"""Disposable, loopback-only TLS edge shared by transport and viewer rehearsals."""

import base64
import hashlib
import os
import secrets
import socket
import ssl
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from monailabel.server.tls import ensure_local_certificate

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def https_edge(directory, backend_port, hostname="localhost"):
    image = os.environ.get("MONAILABEL_TEST_NGINX_IMAGE")
    if not image:
        pytest.skip("Set MONAILABEL_TEST_NGINX_IMAGE to an already installed Nginx image.")

    def docker(*args):
        result = subprocess.run(["docker", *args], text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return result.stdout

    docker("image", "inspect", image)  # Never pull or install.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    certificates = ensure_local_certificate(directory, hostname)
    config = (ROOT / "deploy/demo/https/nginx.conf.template").read_text()
    for key, value in {
        "__PUBLIC_HOST__": hostname,
        "__HTTPS_PORT__": str(port),
        "__BACKEND_PORT__": str(backend_port),
        "__CERTIFICATE_FILE__": "/test-edge/server.crt",
        "__PRIVATE_KEY_FILE__": "/test-edge/server.key",
    }.items():
        config = config.replace(key, value)
    config_path = directory / "nginx.conf"
    config_path.write_text(config)
    controls = [
        "--pull=never",
        "--network",
        "host",
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
        "128m",
        "--cpus",
        "0.5",
        "--tmpfs",
        "/tmp:rw,nosuid,noexec,size=32m",
        "--volume",
        f"{config_path}:/test-nginx.conf:ro",
        "--volume",
        f"{certificates.certificate.parent}:/test-edge:ro",
        "--log-opt",
        "max-size=1m",
        "--log-opt",
        "max-file=1",
        "--entrypoint",
        "nginx",
    ]
    docker("run", "--rm", *controls, image, "-t", "-c", "/test-nginx.conf")
    name = "monailabel-https-test-" + secrets.token_hex(6)
    created = False
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            *controls,
            image,
            "-c",
            "/test-nginx.conf",
            "-g",
            "daemon off;",
        )
        created = True
        context = ssl.create_default_context(cafile=str(certificates.authority))
        for _ in range(100):
            try:
                with context.wrap_socket(
                    socket.create_connection(("127.0.0.1", port), timeout=5),
                    server_hostname=hostname,
                ) as probe:
                    probe.sendall(
                        f"GET /api/auth/status HTTP/1.1\r\nHost: {hostname}:{port}\r\n"
                        "Connection: close\r\n\r\n".encode()
                    )
                    assert b" 200 " in probe.recv(65536).split(b"\r\n", 1)[0]
                break
            except ConnectionRefusedError:
                time.sleep(0.05)
        else:
            pytest.fail("Disposable HTTPS edge did not start")
        certificate = x509.load_pem_x509_certificate(certificates.certificate.read_bytes())
        spki = certificate.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )
        yield SimpleNamespace(
            url=f"https://{hostname}:{port}",
            port=port,
            certificates=certificates,
            context=context,
            spki=base64.b64encode(hashlib.sha256(spki).digest()).decode(),
        )
    finally:
        if created:
            docker("rm", "-f", name)
