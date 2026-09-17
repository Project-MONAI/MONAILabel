"""Encrypted provider credentials; the database stores no plaintext API keys."""

import os
from pathlib import Path

from cryptography.fernet import Fernet

from monailabel.core.errors import DomainError
from monailabel.core.models import Credential, Record
from monailabel.server.storage import Store


class EncryptedCredential(Record):
    ciphertext: str


class Secrets:
    def __init__(self, store: Store, directory: Path):
        self.store = store
        key = directory / "secrets.key"
        if not key.exists():
            with key.open("xb") as stream:
                os.chmod(key, 0o600)
                stream.write(Fernet.generate_key())
        self.cipher = Fernet(key.read_bytes())

    def save(
        self, project_id: str, name: str, value: str, identifier: str | None = None
    ) -> Credential:
        if not name.strip() or not value.strip():
            raise DomainError("A credential name and API key are required.")
        with self.store.transaction() as session:
            if identifier:
                credential = session.get(Credential, identifier)
                if credential.project_id != project_id:
                    raise DomainError("Credential belongs to another project.", status=403)
                credential = credential.model_copy(update={"name": name})
                session.update(credential)
            else:
                credential = Credential(project_id=project_id, name=name)
                session.insert(credential)
            encrypted = EncryptedCredential(
                id=credential.id, ciphertext=self.cipher.encrypt(value.encode()).decode()
            )
            if identifier:
                session.update(encrypted)
            else:
                session.insert(encrypted)
        return credential

    def resolve(self, project_id: str, identifier: str) -> str:
        credential = self.store.get(Credential, identifier)
        if credential.project_id != project_id:
            raise DomainError("Credential belongs to another project.", status=403)
        encrypted = self.store.get(EncryptedCredential, identifier)
        return self.cipher.decrypt(encrypted.ciphertext.encode()).decode()
