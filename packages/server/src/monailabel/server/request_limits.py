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

"""Opt-in pilot ingress bounds, before JSON parsing or application dispatch."""

import asyncio
import tempfile
from threading import Lock

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestLimits:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body: int = 128 * 1024 * 1024,
        auth_body: int = 16 * 1024,
        requests: int = 8,
        waiting_requests: int = 16,
        request_wait: float = 5,
        websockets: int = 16,
        body_timeout: float = 30,
    ):
        self.app = app
        self.max_body, self.auth_body = max_body, auth_body
        self.capacity = {"http": requests, "websocket": websockets}
        self.active = {"http": 0, "websocket": 0}
        self.http_slots = asyncio.Semaphore(requests)
        self.waiting = 0
        self.waiting_requests, self.request_wait = waiting_requests, request_wait
        self.lock = Lock()
        self.body_timeout = body_timeout

    async def admit(self, kind: str) -> bool:
        with self.lock:
            if kind == "websocket":
                if self.active[kind] >= self.capacity[kind]:
                    return False
                self.active[kind] += 1
                return True
            if self.active[kind] + self.waiting >= self.capacity[kind] + self.waiting_requests:
                return False
            self.waiting += 1
        acquired = False
        try:
            # A bounded queue absorbs parallel viewer-module requests without
            # increasing the number of dispatched handlers or upload spools.
            async with asyncio.timeout(self.request_wait):
                await self.http_slots.acquire()
                acquired = True
            return True
        except TimeoutError:
            return False
        finally:
            with self.lock:
                self.waiting -= 1
                if acquired:
                    self.active[kind] += 1

    @staticmethod
    async def reject(scope: Scope, receive: Receive, send: Send, status: int, message: str) -> None:
        headers = {"Cache-Control": "no-store", "Connection": "close"}
        if status == 429:
            headers["Retry-After"] = "5"
        await JSONResponse({"detail": message}, status_code=status, headers=headers)(
            scope, receive, send
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in self.capacity:
            await self.app(scope, receive, send)
            return
        accepted = await self.admit(kind)
        if not accepted:
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 1013})
            else:
                await self.reject(
                    scope, receive, send, 429, "The server is busy. Try again shortly."
                )
            return
        try:
            if kind == "websocket":
                await self.app(scope, receive, send)
            else:
                await self.http(scope, receive, send)
        finally:
            with self.lock:
                self.active[kind] -= 1
                if kind == "http":
                    self.http_slots.release()

    async def http(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.auth_body if scope.get("path", "").startswith("/api/auth/") else self.max_body
        lengths = [
            value for name, value in scope.get("headers", []) if name.lower() == b"content-length"
        ]
        if len(lengths) > 1 or (lengths and (not lengths[0].isdigit() or len(lengths[0]) > 12)):
            await self.reject(scope, receive, send, 400, "Invalid Content-Length.")
            return
        declared = int(lengths[0]) if lengths else None
        if declared is not None and declared > limit:
            await self.reject(
                scope, receive, send, 413, "Request body exceeds this endpoint's limit."
            )
            return
        # Small requests stay in memory; large bodies use an anonymous temporary
        # file. At most `requests * max_body` bytes can be staged at once. No
        # application mutation starts until size and total receive time are checked.
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as spool:
            received = 0
            try:
                async with asyncio.timeout(self.body_timeout):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        block = message.get("body", b"")
                        received += len(block)
                        if received > limit:
                            await self.reject(
                                scope,
                                receive,
                                send,
                                413,
                                "Request body exceeds this endpoint's limit.",
                            )
                            return
                        spool.write(block)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await self.reject(
                    scope, receive, send, 408, "Request body took too long to receive."
                )
                return
            if declared is not None and received != declared:
                await self.reject(
                    scope, receive, send, 400, "Content-Length does not match the body."
                )
                return
            spool.seek(0)
            replayed = False

            async def replay() -> Message:
                nonlocal replayed
                if replayed:
                    return await receive()
                data = spool.read(65536)
                more = spool.tell() < received
                replayed = not more
                return {"type": "http.request", "body": data, "more_body": more}

            await self.app(scope, replay, send)
