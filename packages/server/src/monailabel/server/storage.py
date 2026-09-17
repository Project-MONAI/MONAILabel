"""SQLite transactions and immutable, content-addressed artifacts."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from monailabel.core.errors import NotFound
from monailabel.core.models import Record

T = TypeVar("T", bound=Record)


def referenced_assets(value: object) -> set[str]:
    """Asset identities in records, snapshots, model lineage and job requests."""
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "asset_id" and isinstance(item, str):
                result.add(item)
            elif key in {"asset_ids", "training_assets"} and isinstance(item, list):
                result.update(i for i in item if isinstance(i, str))
            elif isinstance(item, dict | list):
                result.update(referenced_assets(item))
    elif isinstance(value, list):
        for item in value:
            result.update(referenced_assets(item))
    return result


class Session:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def get(self, model: type[T], identifier: str) -> T:
        row = self.connection.execute(
            "SELECT data FROM records WHERE kind=? AND id=?", (model.__name__, identifier)
        ).fetchone()
        if row is None:
            raise NotFound(model.__name__, identifier)
        return model.model_validate_json(row[0])

    def list(self, model: type[T], project_id: str | None = None) -> list[T]:
        query = "SELECT data FROM records WHERE kind=?"
        params: list[str] = [model.__name__]
        if project_id is not None:
            query += " AND project_id=?"
            params.append(project_id)
        query += " ORDER BY created_at,id"
        return [model.model_validate_json(row[0]) for row in self.connection.execute(query, params)]

    def insert(self, record: Record) -> None:
        self.check_deleted(record)
        self.connection.execute(
            "INSERT INTO records(kind,id,project_id,created_at,data) VALUES(?,?,?,?,?)",
            (
                type(record).__name__,
                record.id,
                getattr(record, "project_id", None),
                record.created_at.isoformat(),
                record.model_dump_json(),
            ),
        )

    def update(self, record: Record) -> None:
        self.check_deleted(record)
        cursor = self.connection.execute(
            "UPDATE records SET data=? WHERE kind=? AND id=?",
            (record.model_dump_json(), type(record).__name__, record.id),
        )
        if cursor.rowcount != 1:
            raise NotFound(type(record).__name__, record.id)

    def check_deleted(self, record: Record) -> None:
        # An upload, chat call or cancelled worker may finish after deletion.
        # Tombstones prevent those writes from recreating deleted resources.
        project_id = (
            record.id if type(record).__name__ == "Project" else getattr(record, "project_id", None)
        )
        assets = referenced_assets(record.model_dump())
        if type(record).__name__ == "Asset":
            assets.add(record.id)
        for kind, identifiers in (
            ("Project", [project_id] if project_id else []),
            ("Asset", assets),
        ):
            for identifier in identifiers:
                if self.connection.execute(
                    "SELECT 1 FROM deleted_resources WHERE kind=? AND id=?", (kind, identifier)
                ).fetchone():
                    raise NotFound(kind, identifier)


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as session:
            session.connection.executescript(
                "CREATE TABLE IF NOT EXISTS records ("
                "kind TEXT NOT NULL,id TEXT NOT NULL,project_id TEXT,"
                "created_at TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(kind,id));"
                "CREATE INDEX IF NOT EXISTS records_project ON records(kind,project_id);"
                "CREATE TABLE IF NOT EXISTS deleted_resources ("
                "kind TEXT NOT NULL,id TEXT NOT NULL,PRIMARY KEY(kind,id));"
                "CREATE TABLE IF NOT EXISTS project_changes ("
                "project_id TEXT PRIMARY KEY,version INTEGER NOT NULL);"
                "PRAGMA user_version=3;"
            )
            # Track committed writes from every service, including bulk deletion SQL.
            # This UI change counter is separate from annotation/protocol revisions.
            for event, row in (("insert", "NEW"), ("update", "NEW"), ("delete", "OLD")):
                session.connection.execute(
                    f"CREATE TRIGGER IF NOT EXISTS project_changed_{event} "
                    f"AFTER {event.upper()} ON records "
                    f"WHEN {row}.project_id IS NOT NULL OR {row}.kind='Project' "
                    "BEGIN INSERT INTO project_changes(project_id,version) "
                    f"VALUES(COALESCE({row}.project_id,{row}.id),1) "
                    "ON CONFLICT(project_id) DO UPDATE SET version=version+1; END;"
                )

    def change_version(self, project_id: str) -> int:
        with self.transaction() as session:
            row = session.connection.execute(
                "SELECT version FROM project_changes WHERE project_id=?", (project_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            yield Session(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, model: type[T], identifier: str) -> T:
        with self.transaction() as session:
            return session.get(model, identifier)

    def list(self, model: type[T], project_id: str | None = None) -> list[T]:
        with self.transaction() as session:
            return session.list(model, project_id)


class Artifacts:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}\.(npy|json|bin)", key):
            raise NotFound("Artifact", key)
        return self.root / key[:2] / key

    def put(self, content: bytes, suffix: str = "bin") -> str:
        key = f"{hashlib.sha256(content).hexdigest()}.{suffix}"
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return key

    def read(self, key: str) -> bytes:
        path = self.path(key)
        if not path.is_file():
            raise NotFound("Artifact", key)
        return path.read_bytes()

    def put_file(self, source: Path) -> str:
        """Copy a streamed upload into immutable storage without loading it into RAM."""
        with source.open("rb") as stream:
            key = f"{hashlib.file_digest(stream, 'sha256').hexdigest()}.bin"
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, name = tempfile.mkstemp(dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as output, source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(name, target)
            finally:
                Path(name).unlink(missing_ok=True)
        return key

    def put_array(self, array: NDArray[Any]) -> str:
        # Write and hash incrementally; a CT volume should not need another full byte copy.
        fd, name = tempfile.mkstemp(dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w+b") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                key = f"{digest}.npy"
                target = self.path(key)
                target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.replace(temporary, target)
            return key
        finally:
            temporary.unlink(missing_ok=True)

    def array(self, key: str) -> NDArray[Any]:
        path = self.path(key)
        if not path.is_file():
            raise NotFound("Artifact", key)
        # Read-only mapping preserves immutability and loads just the requested slice.
        return cast(NDArray[Any], np.load(path, mmap_mode="r", allow_pickle=False))

    def put_json(self, value: dict[str, JsonValue]) -> str:
        return self.put(json.dumps(value, sort_keys=True, allow_nan=False).encode(), "json")

    def json(self, key: str) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], json.loads(self.read(key)))
