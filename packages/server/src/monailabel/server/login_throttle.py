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

"""Bound password checks before hashing, without trusting client-supplied headers."""

import hashlib
import math
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock
from time import monotonic

from monailabel.server.demo_access import DemoBusy


class LoginThrottle:
    """Single-process sliding windows, including successful and unknown-user attempts.

    The global window bounds both hashing work and the number of retained keys.
    Rejected requests never allocate a new bucket. A restart clears these short
    windows; this is not a durable account lock or a substitute for edge limits.
    """

    WINDOW = 60
    ACCOUNT_LIMIT = 10
    SOURCE_LIMIT = 30
    GLOBAL_LIMIT = 120
    CONCURRENT_LIMIT = 4

    def __init__(self) -> None:
        self._lock = Lock()
        self._hash_slots = BoundedSemaphore(self.CONCURRENT_LIMIT)
        self._attempts: dict[tuple[str, str], deque[float]] = {}

    @contextmanager
    def attempt(self, username: str, client_address: str) -> Iterator[None]:
        # Retain bounded digests, not account names or client addresses.
        account = hashlib.sha256(username.strip().casefold().encode()).hexdigest()
        source = hashlib.sha256(client_address.encode()).hexdigest()
        limits = (
            (("global", ""), self.GLOBAL_LIMIT),
            (("account", account), self.ACCOUNT_LIMIT),
            (("source", source), self.SOURCE_LIMIT),
        )
        with self._lock:
            now = monotonic()
            for key, attempts in list(self._attempts.items()):
                while attempts and attempts[0] <= now - self.WINDOW:
                    attempts.popleft()
                if not attempts:
                    del self._attempts[key]
            retry = 0
            for key, limit in limits:
                window = self._attempts.get(key)
                if window and len(window) >= limit:
                    retry = max(retry, math.ceil(window[0] + self.WINDOW - now))
            if retry:
                raise DemoBusy(
                    "Too many sign-in attempts. Please wait before trying again.",
                    max(1, retry),
                    code="login_throttled",
                )
            for key, _ in limits:
                self._attempts.setdefault(key, deque()).append(now)
        if not self._hash_slots.acquire(blocking=False):
            raise DemoBusy("Sign-in is busy. Please retry shortly.", 1, code="login_throttled")
        try:
            yield
        finally:
            self._hash_slots.release()
