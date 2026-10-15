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

"""Bounded application-owned spools, not a quota for arbitrary provider writes.

Reserve capacity durably before writing. A crash or failed cleanup keeps its
reservation; no startup sweep guesses whether a worker still owns those files.
"""

import io
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import uuid4

from monailabel.core.models import Job
from monailabel.server.demo_policy import visitor
from monailabel.server.jobs import JobContext
from monailabel.server.storage_quota import exceeded, limit


class SpoolWriter(io.BufferedIOBase):
    def __init__(self, spool: "Spool", name: str, stream: BinaryIO):
        self.spool, self.name, self.stream = spool, name, stream

    def write(self, data: Any) -> int:
        size = memoryview(data).nbytes
        extent = max(self.spool.sizes[self.name], self.stream.tell() + size)
        if self.spool.capacity is not None and (
            sum(self.spool.sizes.values()) - self.spool.sizes[self.name] + extent
            > self.spool.capacity
        ):
            raise exceeded()
        # Charge before the write: an I/O exception can leave partially written bytes.
        self.spool.sizes[self.name] = extent
        return self.stream.write(data)

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.stream.seek(offset, whence)

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def flush(self) -> None:
        if not self.stream.closed:
            self.stream.flush()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.stream.close()

    # Intentionally do not expose fileno: NumPy/Pillow must use checked writes,
    # not bypass accounting with ndarray.tofile or a native encoder's direct fd.


class Spool:
    def __init__(self, directory: Path, capacity: int | None):
        self.directory, self.capacity = directory, capacity
        self.sizes: dict[str, int] = {}
        self.opened: set[str] = set()

    def path(self, name: str) -> Path:
        if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
            raise ValueError("Spool names must be single private filenames.")
        return self.directory / name

    @contextmanager
    def open(self, name: str) -> Iterator[BinaryIO]:
        # Each file is created exactly once. Explicit removal is needed for reuse.
        stream = self.path(name).open("xb")
        self.sizes[name] = 0
        self.opened.add(name)
        try:
            with SpoolWriter(self, name, stream) as writer:
                yield cast(BinaryIO, writer)
        finally:
            self.opened.discard(name)

    def remove(self, name: str) -> None:
        if name in self.opened:
            raise ValueError("Cannot remove an open spool file.")
        if name not in self.sizes:
            raise ValueError("Cannot remove a file not owned by this spool.")
        self.path(name).unlink()
        del self.sizes[name]


class TemporaryReservations:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.ledger = root / ".reservations.sqlite3"
        with self.connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reservations ("
                "id TEXT PRIMARY KEY,owner TEXT,bytes INTEGER NOT NULL)"
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.ledger, timeout=10)
        try:
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def reserve(self, owner: str | None, capacity: int) -> Iterator[Spool]:
        if not 0 < capacity < 2**63:
            raise ValueError("Spool capacity must be positive.")
        identifier = uuid4().hex
        with self.connection() as connection:
            total, count = connection.execute(
                "SELECT COALESCE(SUM(bytes),0),COUNT(*) FROM reservations"
            ).fetchone()
            used, slots = connection.execute(
                "SELECT COALESCE(SUM(bytes),0),COUNT(*) FROM reservations WHERE owner IS ?",
                (owner,),
            ).fetchone()
            if (
                total + capacity > limit("TEMP_TOTAL_BYTES", 8 * 1024**3)
                or count + 1 > limit("TEMP_TOTAL_SPOOLS", 32)
                or (
                    owner is not None
                    and (
                        used + capacity > limit("TEMP_VISITOR_BYTES", 4 * 1024**3)
                        or slots + 1 > limit("TEMP_VISITOR_SPOOLS", 4)
                    )
                )
                or shutil.disk_usage(self.root).free
                < capacity + limit("STORAGE_FREE_BYTES", 1024**3)
            ):
                raise exceeded()
            connection.execute(
                "INSERT INTO reservations VALUES(?,?,?)", (identifier, owner, capacity)
            )
        # Commit the reservation before even creating its private directory.
        directory = self.root / identifier
        directory.mkdir(mode=0o700)
        try:
            yield Spool(directory, capacity)
        finally:
            # This exact fresh directory contains disposable files only. Keep the
            # reservation if removal fails, including after a crash/restart.
            shutil.rmtree(directory)
            with self.connection() as connection:
                connection.execute("DELETE FROM reservations WHERE id=?", (identifier,))


@contextmanager
def job_spool(context: JobContext, setting: str, default: int) -> Iterator[Spool]:
    job = context.store.get(Job, context.job_id)
    leased = visitor(context.store, job.user_id)
    root = context.store.path.parent / "temporary-spools"
    if leased or os.environ.get("MONAILABEL_DEMO_STORAGE_LIMITS") == "1":
        with TemporaryReservations(root).reserve(
            job.user_id if leased else None, limit(setting, default)
        ) as spool:
            yield spool
    else:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="operator-", dir=root) as directory:
            yield Spool(Path(directory), None)
