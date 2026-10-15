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

"""Optional local research telemetry: bounded queue, payload, rate and disk use.

Never pass request bodies, headers, model configuration or image data here.
Free-text redaction is only best-effort: demo participants must use public data.
"""

import json
import os
import queue
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

SECRET = re.compile(
    r"(?i)(?:bearer\s+\S+|\bsk-[\w-]+|\bnvapi-[\w-]+|"
    r"(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[=:]\s*\S+|"
    r"-----BEGIN .*PRIVATE KEY-----)"
)
NOTICE = (
    "Research demo: the host records prompts, assistant replies, workflow actions and errors "
    "for feedback, with bounded local storage and a seven-day retention target. "
    "Use public demo data only; do not enter patient information, passwords or API keys."
)


def text(value: str) -> str:
    # Suppress the whole field, not just one word of a possibly multiword secret.
    if SECRET.search(value):
        return "[omitted: possible credential]"
    return value[:4000] + (" [truncated]" if len(value) > 4000 else "")


class DemoAudit:
    def __init__(self, root: Path, *, enabled: bool = False):
        self.enabled = enabled
        self.root = root
        self.queue: queue.Queue[str] = queue.Queue(maxsize=512)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.dropped = 0
        self.errors = 0
        self.written = 0
        self.window = time.monotonic()
        self.window_count = 0
        self.max_bytes = 5 * 1024**2
        self.files = 4
        self.retention_seconds = 7 * 86400
        self.file_started = 0.0
        self.thread: threading.Thread | None = None
        if enabled:
            try:
                root.mkdir(mode=0o700, parents=True, exist_ok=True)
                root.chmod(0o700)
            except OSError:
                # Report unhealthy telemetry without making annotation unavailable.
                self.errors += 1
            self.thread = threading.Thread(target=self._run, name="demo-audit", daemon=True)
            self.thread.start()

    def record(self, event: str, user_id: str | None = None, **fields: JsonValue) -> None:
        if not self.enabled:
            return
        with self.lock:
            now = time.monotonic()
            if now - self.window >= 1:
                self.window, self.window_count = now, 0
            if self.stop.is_set() or self.window_count >= 100:
                self.dropped += 1
                return
            self.window_count += 1
            clean: dict[str, JsonValue] = {
                "at": datetime.now(UTC).isoformat(),
                "event": event[:80],
                "user_id": user_id,
            }
            # Scalars only; no arbitrary recursive objects or raw tool arguments.
            for key, value in list(fields.items())[:16]:
                if isinstance(value, str):
                    clean[key] = text(value)
                elif value is None or isinstance(value, bool | int | float):
                    clean[key] = value
            line = json.dumps(clean, ensure_ascii=True) + "\n"
            if len(line.encode()) > 16384:
                self.dropped += 1
                return
            try:
                self.queue.put_nowait(line)
            except queue.Full:
                self.dropped += 1

    def _path(self, index: int) -> Path:
        return self.root / f"events.{index}.jsonl"

    def _prune(self) -> None:
        for index in range(self.files):
            path = self._path(index)
            if path.exists():
                try:
                    with path.open() as stream:
                        first = json.loads(stream.readline(16385))
                    started = datetime.fromisoformat(first["at"]).timestamp()
                except (ValueError, KeyError):
                    started = path.stat().st_mtime
                if time.time() - started > self.retention_seconds:
                    path.unlink()
                elif index == 0:
                    self.file_started = started

    def _write(self, line: str) -> None:
        path = self._path(0)
        if path.exists() and (
            path.stat().st_size + len(line.encode()) > self.max_bytes
            or time.time() - self.file_started >= 86400
        ):
            self._path(self.files - 1).unlink(missing_ok=True)
            for index in range(self.files - 2, -1, -1):
                previous = self._path(index)
                if previous.exists():
                    previous.replace(self._path(index + 1))
        if not path.exists():
            self.file_started = time.time()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(line)
        with self.lock:
            self.written += 1

    def _run(self) -> None:
        next_prune = 0.0
        while not self.stop.is_set() or not self.queue.empty():
            try:
                if time.monotonic() >= next_prune:
                    self._prune()
                    next_prune = time.monotonic() + 60
                line = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            except OSError:
                with self.lock:
                    self.errors += 1
                next_prune = time.monotonic() + 60
                continue
            try:
                self._write(line)
            except OSError:
                with self.lock:
                    self.errors += 1
                    self.dropped += 1
            finally:
                self.queue.task_done()

    def status(self) -> dict[str, JsonValue]:
        with self.lock:
            return {
                "enabled": self.enabled,
                "queued": self.queue.qsize(),
                "written": self.written,
                "dropped": self.dropped,
                "write_errors": self.errors,
                "max_disk_bytes": self.max_bytes * self.files,
                "retention_days": 7,
            }

    def close(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
