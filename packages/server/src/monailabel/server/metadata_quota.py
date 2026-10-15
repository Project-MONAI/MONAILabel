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

"""Transactional, cumulative metadata allocation for leased demo visitors.

Record allocations and claims are high-water marks, including deleted records.
Deletion/reset cannot recycle an allowance into unlimited ledger, tombstone or
project-change growth. These logical budgets are not filesystem/SQLite quotas.
"""

import json
import os
import sqlite3

from monailabel.core.errors import DomainError
from monailabel.core.models import Record
from monailabel.server.demo_context import actor
from monailabel.server.storage_quota import exceeded, limit

# Reserve space at creation for bounded lifecycle/status changes. In particular,
# even a full visitor budget must allow a queued/running job to become terminal.
RECORD_RESERVE = 512
JOB_RESERVE = 64 * 1024
IDENTITY_KINDS = {"User", "Password", "DemoAccess", "DemoVisit"}


class MetadataQuota:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def initialize(self) -> None:
        """Called in one schema transaction; never changes existing user records."""
        connection = self.connection
        for statement in (
            "CREATE TABLE IF NOT EXISTS metadata_objects ("
            "kind TEXT NOT NULL,id TEXT NOT NULL,bytes INTEGER NOT NULL,PRIMARY KEY(kind,id))",
            "CREATE TABLE IF NOT EXISTS metadata_claims ("
            "owner TEXT NOT NULL,kind TEXT NOT NULL,id TEXT NOT NULL,bytes INTEGER NOT NULL,"
            "PRIMARY KEY(owner,kind,id))",
            "CREATE INDEX IF NOT EXISTS metadata_claimed_object ON metadata_claims(kind,id)",
            "CREATE TABLE IF NOT EXISTS metadata_usage ("
            "id INTEGER PRIMARY KEY,bytes INTEGER NOT NULL,objects INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS metadata_initialized (id INTEGER PRIMARY KEY)",
            "INSERT OR IGNORE INTO metadata_usage VALUES(1,0,0)",
            "CREATE TRIGGER IF NOT EXISTS metadata_object_insert AFTER INSERT ON metadata_objects "
            "BEGIN UPDATE metadata_usage SET bytes=bytes+NEW.bytes,objects=objects+1 "
            "WHERE id=1; END",
            "CREATE TRIGGER IF NOT EXISTS metadata_object_update AFTER UPDATE ON metadata_objects "
            "BEGIN UPDATE metadata_usage SET bytes=bytes+NEW.bytes-OLD.bytes WHERE id=1; "
            "UPDATE metadata_claims SET bytes=MAX(bytes,NEW.bytes) "
            "WHERE kind=NEW.kind AND id=NEW.id; END",
        ):
            connection.execute(statement)
        # SQL triggers account for trusted internal SQL too, not just Session writes.
        # New objects receive headroom; updates consume it before needing more space.
        for event in ("INSERT", "UPDATE"):
            reserve = (
                f"+CASE WHEN NEW.kind='Job' THEN {JOB_RESERVE} ELSE {RECORD_RESERVE} END"
                if event == "INSERT"
                else ""
            )
            connection.execute(
                f"CREATE TRIGGER IF NOT EXISTS metadata_record_{event.lower()} "
                f"AFTER {event} ON records BEGIN "
                "INSERT INTO metadata_objects VALUES(NEW.kind,NEW.id,"
                f"length(CAST(NEW.data AS BLOB)){reserve}) "
                "ON CONFLICT(kind,id) DO UPDATE SET bytes=MAX(bytes,excluded.bytes); END"
            )
        if not connection.execute("SELECT 1 FROM metadata_initialized").fetchone():
            connection.execute(
                "INSERT INTO metadata_objects SELECT kind,id,length(CAST(data AS BLOB))+"
                f"CASE WHEN kind='Job' THEN {JOB_RESERVE} ELSE {RECORD_RESERVE} END "
                "FROM records WHERE 1 ON CONFLICT(kind,id) DO NOTHING"
            )
            self.backfill()
            connection.execute("INSERT INTO metadata_initialized VALUES(1)")

    def is_visitor(self, identifier: str | None) -> bool:
        return bool(
            identifier
            and self.connection.execute(
                "SELECT 1 FROM records WHERE kind='DemoAccess' AND id=?", (identifier,)
            ).fetchone()
        )

    def owner(self, kind: str, identifier: str, content: dict[str, object]) -> str | None:
        if kind == "DemoAccess":
            return identifier
        current = actor.get()
        if self.is_visitor(current):
            return current
        candidate = identifier if kind in IDENTITY_KINDS else content.get("user_id")
        # Progress/cleanup and authentication sometimes execute outside an HTTP
        # actor scope. Recover only server-record ownership, never a request flag.
        if kind in {"JobLog", "ToolExecution"}:
            parent, field = (
                ("Job", "job_id") if kind == "JobLog" else ("Conversation", "conversation_id")
            )
            row = self.connection.execute(
                "SELECT json_extract(data,'$.user_id') FROM records WHERE kind=? AND id=?",
                (parent, content.get(field)),
            ).fetchone()
            candidate = row[0] if row else None
        return candidate if isinstance(candidate, str) and self.is_visitor(candidate) else None

    def usage(self, owner: str | None = None) -> tuple[int, int]:
        if owner is None:
            row = self.connection.execute(
                "SELECT bytes,objects FROM metadata_usage WHERE id=1"
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COALESCE(SUM(bytes),0),COUNT(*) FROM metadata_claims WHERE owner=?",
                (owner,),
            ).fetchone()
        return int(row[0]), int(row[1])

    def check(self, record: Record, data: str, *, insert: bool) -> str | None:
        kind = type(record).__name__
        owner = self.owner(kind, record.id, record.model_dump())
        row = self.connection.execute(
            "SELECT bytes FROM metadata_objects WHERE kind=? AND id=?", (kind, record.id)
        ).fetchone()
        previous = int(row[0]) if row else 0
        reserve = JOB_RESERVE if kind == "Job" else RECORD_RESERVE
        allocation = max(previous, len(data.encode()) + (reserve if insert else 0))
        if owner is not None or os.environ.get("MONAILABEL_DEMO_STORAGE_LIMITS") == "1":
            delta = allocation - previous
            if delta or row is None:
                used_bytes, used_objects = self.usage()
                if used_bytes + delta > limit(
                    "METADATA_TOTAL_BYTES", 512 * 1024**2
                ) or used_objects + (row is None) > limit("METADATA_TOTAL_RECORDS", 200_000):
                    raise exceeded()
            if owner is not None:
                self.check_claim(owner, kind, record.id, allocation)
        return owner

    def check_claim(self, owner: str, kind: str, identifier: str, allocation: int) -> None:
        row = self.connection.execute(
            "SELECT bytes FROM metadata_claims WHERE owner=? AND kind=? AND id=?",
            (owner, kind, identifier),
        ).fetchone()
        delta = max(0, allocation - int(row[0])) if row else allocation
        if not delta and row:
            return
        used_bytes, used_objects = self.usage(owner)
        if (
            used_bytes + delta > limit("METADATA_BYTES", 64 * 1024**2)
            or used_objects + (row is None) > limit("METADATA_RECORDS", 20_000)
            or (
                row is None
                and self.connection.execute("SELECT COUNT(*) FROM metadata_claims").fetchone()[0]
                + 1
                > limit("METADATA_TOTAL_CLAIMS", 400_000)
            )
        ):
            raise exceeded()

    def claim(self, owner: str | None, kind: str, identifier: str) -> None:
        if owner is not None:
            self.connection.execute(
                "INSERT INTO metadata_claims SELECT ?,kind,id,bytes FROM metadata_objects "
                "WHERE kind=? AND id=? ON CONFLICT(owner,kind,id) "
                "DO UPDATE SET bytes=MAX(bytes,excluded.bytes)",
                (owner, kind, identifier),
            )

    def backfill(self, owner: str | None = None) -> None:
        """One-time migration, or a new invitation/signup in its creation transaction."""
        connection = self.connection
        visitors = {
            row[0] for row in connection.execute("SELECT id FROM records WHERE kind='DemoAccess'")
        }
        if owner is not None:
            visitors.intersection_update({owner})
        projects: dict[str, set[str]] = {}
        for (data,) in connection.execute("SELECT data FROM records WHERE kind='Membership'"):
            member = json.loads(data)
            if member["user_id"] in visitors:
                projects.setdefault(member["project_id"], set()).add(member["user_id"])
        rows = connection.execute("SELECT kind,id,project_id,data FROM records").fetchall()
        for kind, identifier, project, data in rows:
            content = json.loads(data)
            owners = projects.get(identifier if kind == "Project" else project, set()).copy()
            candidate = identifier if kind in IDENTITY_KINDS else content.get("user_id")
            if candidate in visitors:
                owners.add(candidate)
            for visitor in owners:
                exists = connection.execute(
                    "SELECT 1 FROM metadata_claims WHERE owner=? AND kind=? AND id=?",
                    (visitor, kind, identifier),
                ).fetchone()
                if exists:
                    continue
                if owner is not None:
                    allocation = connection.execute(
                        "SELECT bytes FROM metadata_objects WHERE kind=? AND id=?",
                        (kind, identifier),
                    ).fetchone()[0]
                    self.check_claim(visitor, kind, identifier, allocation)
                elif connection.execute("SELECT COUNT(*) FROM metadata_claims").fetchone()[
                    0
                ] >= limit("METADATA_TOTAL_CLAIMS", 400_000):
                    raise DomainError(
                        "Existing demo data exceeds the metadata claim limit. "
                        "The host must increase it before startup; existing data is preserved.",
                        status=503,
                    )
                self.claim(visitor, kind, identifier)
