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

"""Durable artifact reservations. Never hold a workspace writer transaction here.

The separate ledger commits reservations before filesystem writes. A private file
lock serializes writers across processes; interrupted reservations remain charged
until startup reconciliation under the exclusive workspace/server lock.
"""

import json
import os
import re
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from monailabel.core.errors import DomainError
from monailabel.server.demo_context import actor

KEY = re.compile(r"[0-9a-f]{64}\.(npy|json|bin)")


def limit(name: str, default: int) -> int:
    try:
        value = int(os.environ.get("MONAILABEL_DEMO_" + name, str(default)))
    except ValueError:
        value = 0
    if not 0 < value < 2**63:
        raise DomainError("The host must configure positive storage limits.", status=503)
    return value


def exceeded() -> DomainError:
    return DomainError(
        "Demo storage is full. Existing work is preserved; "
        "ask the host for space before saving more.",
        code="storage_quota_exceeded",
        status=507,
    )


class ArtifactQuota:
    def __init__(self, root: Path, database: Path):
        self.root, self.database = root, database.resolve()
        self.ledger = root / ".quota.sqlite3"
        self.lock = FileLock(str(root / ".quota.lock"), timeout=30)
        with self.lock, self.connection() as connection:
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS objects ("
                "key TEXT PRIMARY KEY,bytes INTEGER NOT NULL,state TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS claims ("
                "owner TEXT NOT NULL,key TEXT NOT NULL,PRIMARY KEY(owner,key),"
                "FOREIGN KEY(key) REFERENCES objects(key) ON DELETE CASCADE);"
                "CREATE TABLE IF NOT EXISTS initialized (id INTEGER PRIMARY KEY);"
            )
            if not connection.execute("SELECT 1 FROM initialized").fetchone():
                # Count pre-existing host files too; enabling a quota never deletes them.
                for path in self.files():
                    connection.execute(
                        "INSERT INTO objects VALUES(?,?,'ready')",
                        (path.name, path.stat().st_size),
                    )
                connection.execute("INSERT INTO initialized VALUES(1)")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.ledger, timeout=30)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def files(self) -> Iterator[Path]:
        for path in self.root.glob("[0-9a-f][0-9a-f]/*"):
            if KEY.fullmatch(path.name) and path.is_file():
                yield path

    @contextmanager
    def workspace(self) -> Iterator[sqlite3.Connection]:
        # A WAL reader can coexist with a caller's publishing transaction. Store.get
        # would start another BEGIN IMMEDIATE writer and could deadlock that caller.
        connection = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            yield connection
        finally:
            connection.close()

    def owner(self) -> str | None:
        identifier = actor.get()
        if identifier is None:
            return None
        with self.workspace() as connection:
            row = connection.execute(
                "SELECT 1 FROM records WHERE kind='DemoAccess' AND id=?", (identifier,)
            ).fetchone()
        return identifier if row else None

    @staticmethod
    def usage(connection: sqlite3.Connection, owner: str | None = None) -> tuple[int, int]:
        query = "SELECT COALESCE(SUM(bytes),0),COUNT(*) FROM objects"
        parameters: tuple[str, ...] = ()
        if owner is not None:
            query += " JOIN claims USING(key) WHERE owner=?"
            parameters = (owner,)
        row = connection.execute(query, parameters).fetchone()
        return int(row[0]), int(row[1])

    def check(
        self, connection: sqlite3.Connection, owner: str | None, key: str, size: int, new: bool
    ) -> None:
        if owner is None and os.environ.get("MONAILABEL_DEMO_STORAGE_LIMITS") != "1":
            return
        if new:
            total_bytes, total_objects = self.usage(connection)
            if total_bytes + size > limit(
                "ARTIFACT_TOTAL_BYTES", 64 * 1024**3
            ) or total_objects + 1 > limit("ARTIFACT_TOTAL_OBJECTS", 200_000):
                raise exceeded()
        if (
            owner is not None
            and not connection.execute(
                "SELECT 1 FROM claims WHERE owner=? AND key=?", (owner, key)
            ).fetchone()
        ):
            used_bytes, used_objects = self.usage(connection, owner)
            if (
                used_bytes + size > limit("ARTIFACT_BYTES", 8 * 1024**3)
                or used_objects + 1 > limit("ARTIFACT_OBJECTS", 10_000)
                or connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0] + 1
                > limit("ARTIFACT_TOTAL_CLAIMS", 200_000)
            ):
                raise exceeded()

    @contextmanager
    def reserve(self, target: Path, size: int) -> Iterator[Path]:
        owner = self.owner()
        key = target.name
        if not KEY.fullmatch(key) or target != self.root / key[:2] / key or size < 0:
            raise ValueError("Invalid artifact reservation")
        temporary = target.with_name(".pending-" + key)
        try:
            self.lock.acquire()
        except Timeout as error:
            raise DomainError("Storage is busy. Retry shortly.", status=429) from error
        try:
            with self.connection() as connection:
                row = connection.execute("SELECT bytes FROM objects WHERE key=?", (key,)).fetchone()
                if row and row[0] != size:
                    raise DomainError("Artifact size differs from the immutable storage record.")
                new = row is None
                claimed = bool(
                    owner
                    and connection.execute(
                        "SELECT 1 FROM claims WHERE owner=? AND key=?", (owner, key)
                    ).fetchone()
                )
                self.check(connection, owner, key, size, new)
                if (
                    (owner is not None or os.environ.get("MONAILABEL_DEMO_STORAGE_LIMITS") == "1")
                    and not target.exists()
                    and shutil.disk_usage(self.root).free
                    < size + limit("STORAGE_FREE_BYTES", 1024**3)
                ):
                    raise exceeded()
                if new:
                    connection.execute("INSERT INTO objects VALUES(?,?,'pending')", (key, size))
                if owner:
                    connection.execute("INSERT OR IGNORE INTO claims VALUES(?,?)", (owner, key))
            # Reservation is committed and includes any previous interrupted partial
            # write at this exact private path. No other writer can own it now.
            try:
                yield temporary
                if not target.is_file() or target.stat().st_size != size:
                    raise DomainError("Artifact publication did not preserve its reserved size.")
            except BaseException:
                # Never release a reservation while its partial bytes remain on disk.
                temporary.unlink(missing_ok=True)
                with self.connection() as connection:
                    if not target.exists():
                        if new:
                            connection.execute("DELETE FROM objects WHERE key=?", (key,))
                        elif owner and not claimed:
                            connection.execute(
                                "DELETE FROM claims WHERE owner=? AND key=?", (owner, key)
                            )
                raise
            else:
                with self.connection() as connection:
                    connection.execute("UPDATE objects SET state='ready' WHERE key=?", (key,))
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            self.lock.release()

    def reconcile(self) -> None:
        """Startup only: caller holds the server lock and has not started workers."""
        with self.lock, self.connection() as connection:
            existing = {path.name: path for path in self.files()}
            for (key,) in connection.execute("SELECT key FROM objects").fetchall():
                if not KEY.fullmatch(key):
                    raise DomainError("Invalid storage ledger entry.", status=503)
                # Only our exact, ledger-owned partial paths are removed, never drafts.
                (self.root / key[:2] / (".pending-" + key)).unlink(missing_ok=True)
                if key not in existing:
                    connection.execute("DELETE FROM objects WHERE key=?", (key,))
            for key, path in existing.items():
                connection.execute(
                    "INSERT INTO objects VALUES(?,?,'ready') ON CONFLICT(key) "
                    "DO UPDATE SET bytes=excluded.bytes,state='ready'",
                    (key, path.stat().st_size),
                )
            self.backfill_claims(connection)

    def backfill_claims(self, connection: sqlite3.Connection) -> None:
        """Charge existing visitors for retained project data, following JSON manifests."""
        with self.workspace() as workspace:
            visitors = {
                r[0] for r in workspace.execute("SELECT id FROM records WHERE kind='DemoAccess'")
            }
            projects: dict[str, set[str]] = {}
            for (data,) in workspace.execute("SELECT data FROM records WHERE kind='Membership'"):
                member = json.loads(data)
                if member["user_id"] in visitors:
                    projects.setdefault(member["project_id"], set()).add(member["user_id"])
            rows = workspace.execute("SELECT kind,id,project_id,data FROM records").fetchall()
        pending: dict[str, set[str]] = {owner: set() for owner in visitors}

        def keys(value: object) -> set[str]:
            if isinstance(value, str):
                return {value} if KEY.fullmatch(value) else set()
            if isinstance(value, dict):
                return set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        for kind, identifier, project, data in rows:
            content = json.loads(data)
            owners = projects.get(identifier if kind == "Project" else project, set()).copy()
            if content.get("user_id") in visitors:
                owners.add(content["user_id"])
            for owner in owners:
                pending[owner].update(keys(content))
        manifests: dict[str, set[str]] = {}
        claims = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        for owner, remaining in pending.items():
            visited: set[str] = set()
            while remaining:
                key = remaining.pop()
                if key in visited:
                    continue
                visited.add(key)
                if not connection.execute("SELECT 1 FROM objects WHERE key=?", (key,)).fetchone():
                    continue
                if not connection.execute(
                    "SELECT 1 FROM claims WHERE owner=? AND key=?", (owner, key)
                ).fetchone():
                    if claims >= limit("ARTIFACT_TOTAL_CLAIMS", 200_000):
                        raise DomainError(
                            "Existing demo data exceeds the storage claim limit. "
                            "The host must increase it before startup; existing data is preserved.",
                            status=503,
                        )
                    connection.execute("INSERT INTO claims VALUES(?,?)", (owner, key))
                    claims += 1
                if key.endswith(".json"):
                    if key not in manifests:
                        manifests[key] = keys(json.loads((self.root / key[:2] / key).read_bytes()))
                    remaining.update(manifests[key] - visited)
