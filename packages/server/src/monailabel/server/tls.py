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

"""Workspace-local HTTPS certificates; never modifies the device's trust store."""

import os
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from filelock import FileLock

from monailabel.server.network import allowed_hosts


@dataclass(frozen=True)
class LocalCertificate:
    certificate: Path
    key: Path
    authority: Path


def _write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        os.chmod(temporary, 0o600)
        stream.write(content)
    temporary.replace(path)


def _private_key(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _key_usage(*, authority: bool) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=True,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=authority,
        crl_sign=authority,
        encipher_only=False,
        decipher_only=False,
    )


def _names(host: str) -> x509.SubjectAlternativeName:
    names: set[x509.GeneralName] = set()
    for name in {*allowed_hosts(), host, "localhost", "127.0.0.1", "::1"} - {"testserver"}:
        name = name.strip("[]").split("%", 1)[0]
        try:
            address = ip_address(name)
        except ValueError:
            if "*" not in name:
                names.add(x509.DNSName(name.encode("idna").decode("ascii")))
        else:
            if not address.is_unspecified:
                names.add(x509.IPAddress(address))
    return x509.SubjectAlternativeName(sorted(names, key=lambda name: str(name.value)))


def _builder(
    subject: x509.Name, key: ec.EllipticCurvePrivateKey, days: int
) -> x509.CertificateBuilder:
    now = datetime.now(UTC)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False)
    )


def ensure_local_certificate(workspace: Path, host: str) -> LocalCertificate:
    directory = workspace / ".tls"
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    paths = LocalCertificate(
        directory / "server.crt", directory / "server.key", directory / "ca.crt"
    )
    ca_key_path = directory / "ca.key"
    with FileLock(str(directory / "certificates.lock")):
        if paths.authority.exists() != ca_key_path.exists():
            raise ValueError(
                "Local HTTPS authority is incomplete; "
                "restore .tls/ca.crt and .tls/ca.key from backup."
            )
        if paths.authority.exists():
            authority = x509.load_pem_x509_certificate(paths.authority.read_bytes())
            ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
            if (
                not isinstance(ca_key, ec.EllipticCurvePrivateKey)
                or ca_key.public_key() != authority.public_key()
            ):
                raise ValueError("Local HTTPS authority and private key do not match.")
            if authority.not_valid_after_utc < datetime.now(UTC) + timedelta(days=31):
                raise ValueError(
                    "Local HTTPS authority is expiring; remove .tls to regenerate it "
                    "and trust the new ca.crt on your devices."
                )
        else:
            ca_key = ec.generate_private_key(ec.SECP256R1())
            subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MONAI Label local CA")])
            authority = (
                _builder(subject, ca_key, 3650)
                .issuer_name(subject)
                .add_extension(x509.BasicConstraints(ca=True, path_length=0), True)
                .add_extension(_key_usage(authority=True), True)
                .sign(ca_key, hashes.SHA256())
            )
            _write(ca_key_path, _private_key(ca_key))
            _write(paths.authority, authority.public_bytes(serialization.Encoding.PEM))
        names = _names(host)
        if paths.certificate.exists() and paths.key.exists():
            certificate = x509.load_pem_x509_certificate(paths.certificate.read_bytes())
            certificate.verify_directly_issued_by(authority)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(paths.certificate, paths.key)
            if certificate.not_valid_after_utc > datetime.now(UTC) + timedelta(days=30) and set(
                names
            ).issubset(
                certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            ):
                return paths
        key = ec.generate_private_key(ec.SECP256R1())
        certificate = (
            _builder(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MONAI Label")]), key, 365)
            .issuer_name(authority.subject)
            .add_extension(names, False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(_key_usage(authority=False), True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False
            )
            .sign(ca_key, hashes.SHA256())
        )
        _write(paths.key, _private_key(key))
        _write(paths.certificate, certificate.public_bytes(serialization.Encoding.PEM))
        return paths
