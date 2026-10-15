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

"""Bound authenticated HTTP request rates by account, not renewable tokens."""

import math
from threading import Lock
from time import monotonic

from monailabel.server.demo_access import DemoBusy


class RequestThrottle:
    RATE = 5  # Refilled tokens per second.
    BURST = 60
    READ_RATE = 30
    READ_BURST = 256
    MAX_BUCKETS = 1024

    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.lock = Lock()
        self.accounts: dict[tuple[str, bool], tuple[float, float]] = {}

    def check(self, user_id: str, *, read: bool = False) -> None:
        if not self.enabled:
            return
        with self.lock:
            now = monotonic()
            # A fully refilled inactive bucket can be removed without forgiving debt.
            for key, (_, updated) in list(self.accounts.items()):
                idle = self.READ_BURST / self.READ_RATE if key[1] else self.BURST / self.RATE
                if now - updated >= idle:
                    del self.accounts[key]
            key = (user_id, read)
            rate, burst = (self.READ_RATE, self.READ_BURST) if read else (self.RATE, self.BURST)
            if key not in self.accounts and len(self.accounts) >= self.MAX_BUCKETS:
                raise DemoBusy("Request capacity is busy. Try again shortly.", 1)
            tokens, previous = self.accounts.get(key, (float(burst), now))
            tokens = min(burst, tokens + max(0, now - previous) * rate)
            if tokens < 1:
                raise DemoBusy(
                    "Too many requests for this account. Please wait before retrying.",
                    max(1, math.ceil((1 - tokens) / rate)),
                    code="request_throttled",
                )
            self.accounts[key] = (tokens - 1, now)
