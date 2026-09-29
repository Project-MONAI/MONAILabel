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

"""Local HTTPS certificate lifecycle, verified transport and command-line wiring."""

import json
import ssl
import sys
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

from monailabel.server.healthcheck import check
from monailabel.server.main import main
from monailabel.server.tls import ensure_local_certificate


@pytest.fixture
def certificates(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "MONAILABEL_ALLOWED_HOSTS", "localhost,127.0.0.1,192.0.2.20,[::1],example.test"
    )
    return ensure_local_certificate(tmp_path, "0.0.0.0")


def test_certificate_reuse_names_and_private_keys(tmp_path, certificates):
    before = {
        path: path.read_bytes()
        for path in (certificates.certificate, certificates.key, certificates.authority)
    }
    ensure_local_certificate(tmp_path, "0.0.0.0")
    assert all(path.read_bytes() == data for path, data in before.items())
    assert certificates.key.stat().st_mode & 0o777 == 0o600
    assert certificates.key.parent.stat().st_mode & 0o777 == 0o700
    cert = x509.load_pem_x509_certificate(certificates.certificate.read_bytes())
    names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert {str(item.value) for item in names} == {
        "localhost",
        "127.0.0.1",
        "::1",
        "192.0.2.20",
        "example.test",
    }
    cert.verify_directly_issued_by(
        x509.load_pem_x509_certificate(certificates.authority.read_bytes())
    )


def test_new_address_renews_leaf_without_changing_trust(tmp_path, certificates):
    authority = certificates.authority.read_bytes()
    old = certificates.certificate.read_bytes()
    ensure_local_certificate(tmp_path, "192.0.2.21")
    assert certificates.authority.read_bytes() == authority
    assert certificates.certificate.read_bytes() != old


def test_expiring_leaf_renews_without_changing_trust(tmp_path, certificates):
    authority = x509.load_pem_x509_certificate(certificates.authority.read_bytes())
    ca_key = serialization.load_pem_private_key(
        (certificates.key.parent / "ca.key").read_bytes(), None
    )
    old = x509.load_pem_x509_certificate(certificates.certificate.read_bytes())
    builder = (
        x509.CertificateBuilder()
        .subject_name(old.subject)
        .issuer_name(authority.subject)
        .public_key(old.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(days=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
    )
    for extension in old.extensions:
        builder = builder.add_extension(extension.value, extension.critical)
    certificates.certificate.write_bytes(
        builder.sign(ca_key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM)
    )
    ensure_local_certificate(tmp_path, "0.0.0.0")
    renewed = x509.load_pem_x509_certificate(certificates.certificate.read_bytes())
    assert renewed.not_valid_after_utc > datetime.now(UTC) + timedelta(days=300)
    renewed.verify_directly_issued_by(authority)


def test_incomplete_authority_is_not_silently_replaced(tmp_path, certificates):
    certificates.authority.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        ensure_local_certificate(tmp_path, "0.0.0.0")


@pytest.mark.parametrize("https", [False, True])
def test_verified_https_and_container_health(tmp_path, monkeypatch, certificates, https):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/auth/status"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"setup_required": true}')

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        if https:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certificates.certificate, certificates.key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setenv("MONAILABEL_DATA_DIR", str(tmp_path))
            monkeypatch.setenv("MONAILABEL_HEALTH_PORT", str(server.server_port))
            check()
            if https:
                url = f"https://127.0.0.1:{server.server_port}/api/auth/status"
                with pytest.raises(urllib.error.URLError, match="CERTIFICATE_VERIFY_FAILED"):
                    urllib.request.urlopen(url)
                with urllib.request.urlopen(
                    url, context=ssl.create_default_context(cafile=str(certificates.authority))
                ) as response:
                    assert json.load(response)["setup_required"]
        finally:
            server.shutdown()
            thread.join()


def test_https_cli_preserves_port_and_passes_public_trust(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monailabel",
            "--https",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
            "--data-dir",
            str(tmp_path),
        ],
    )
    app = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setattr("monailabel.server.main.create_app", lambda *args, **kwargs: app)
    monkeypatch.setattr("monailabel.server.main.configure_workspace", lambda directory: directory)
    calls = []
    monkeypatch.setattr(
        "monailabel.server.main.uvicorn.run", lambda app, **kwargs: calls.append(kwargs)
    )
    main()
    assert calls[0]["port"] == 8123 and calls[0]["host"] == "0.0.0.0"
    assert calls[0]["ssl_certfile"] == str(tmp_path / ".tls/server.crt")
    assert app.state.direct_tls
    assert "BEGIN CERTIFICATE" in app.state.local_ca_certificate
    assert "PRIVATE KEY" not in app.state.local_ca_certificate
    assert "Trust" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flags",
    [["--ssl-certfile", "a.crt"], ["--https", "--ssl-certfile", "a.crt", "--ssl-keyfile", "a.key"]],
)
def test_https_rejects_ambiguous_or_incomplete_flags(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["monailabel", *flags])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
