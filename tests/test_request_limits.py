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

import asyncio
import json
import tempfile

import pytest
from chat_fixture import ScriptedChat
from fastapi.testclient import TestClient

from monailabel.server.app import create_app
from monailabel.server.request_limits import RequestLimits


def scope(path="/api/data", headers=()):
    return {"type": "http", "path": path, "headers": list(headers), "method": "POST"}


def test_header_chunked_auth_and_mismatched_lengths_are_checked_before_dispatch():
    async def run():
        calls = []

        async def app(scope, receive, send):
            calls.append(scope)

        guard = RequestLimits(app, max_body=32, auth_body=8)
        cases = [
            (scope(headers=[(b"content-length", b"33")]), [], 413),
            (scope(headers=[(b"content-length", b"-1")]), [], 400),
            (scope(headers=[(b"content-length", b"1"), (b"content-length", b"1")]), [], 400),
            (scope(), [b"x" * 16, b"x" * 17], 413),
            (scope("/api/auth/login"), [b"x" * 9], 413),
            (scope(headers=[(b"content-length", b"5")]), [b"xx"], 400),
        ]
        for request, parts, expected in cases:
            messages = [
                {"type": "http.request", "body": block, "more_body": i < len(parts) - 1}
                for i, block in enumerate(parts)
            ]
            replies = []

            async def receive(messages=messages):
                return messages.pop(0)

            async def send(message, replies=replies):
                replies.append(message)

            await guard(request, receive, send)
            assert replies[0]["status"] == expected
            assert (b"cache-control", b"no-store") in replies[0]["headers"]
            assert not guard.active["http"]
        assert not calls

    asyncio.run(run())


def test_stream_replays_exact_bytes_and_disconnect_and_timeout_do_not_dispatch():
    async def run():
        original = b"abcd" * 300000  # Exercises disk spool and bounded replay chunks.
        replayed = bytearray()

        async def app(scope, receive, send):
            while True:
                message = await receive()
                assert len(message["body"]) <= 65536
                replayed.extend(message["body"])
                if not message["more_body"]:
                    break

        guard = RequestLimits(app, body_timeout=0.02)
        pending = [{"type": "http.request", "body": original, "more_body": False}]
        output = []

        async def receive():
            return pending.pop(0)

        async def send(message):
            output.append(message)

        await guard(scope(), receive, send)
        assert replayed == original and not output
        replayed.clear()
        pending.append({"type": "http.disconnect"})
        await guard(scope(), receive, send)
        assert not replayed and not output

        async def slow():
            await asyncio.sleep(1)
            raise AssertionError("Timeout should cancel the read")

        await guard(scope(), slow, send)
        assert output[0]["status"] == 408 and not replayed
        assert not guard.active["http"]

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["http", "websocket"])
def test_concurrency_is_bounded_and_cancellation_releases_slots(kind):
    async def run():
        finish = asyncio.Event()
        started = asyncio.Event()

        async def app(scope, receive, send):
            started.set()
            await finish.wait()

        guard = RequestLimits(app, requests=1, waiting_requests=0, websockets=1)
        request = scope() | {"type": kind}
        replies = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            replies.append(message)

        first = asyncio.create_task(guard(request, receive, send))
        await started.wait()
        await guard(request, receive, send)
        if kind == "http":
            assert replies[0]["status"] == 429
            assert (b"retry-after", b"5") in replies[0]["headers"]
        else:
            assert replies == [{"type": "websocket.close", "code": 1013}]
        assert guard.active[kind] == 1
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not guard.active[kind]
        finish.set()
        await guard(request, receive, send)
        assert not guard.active[kind]

    asyncio.run(run())


def test_waiting_queue_is_bounded_and_does_not_read_or_dispatch_until_admitted():
    async def run():
        release = asyncio.Event()
        started = asyncio.Event()
        reads = []

        async def app(scope, receive, send):
            started.set()
            await release.wait()

        guard = RequestLimits(app, requests=1, waiting_requests=1, request_wait=1)
        replies = []

        async def receive():
            reads.append(True)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            replies.append(message)

        first = asyncio.create_task(guard(scope(), receive, send))
        await started.wait()
        second = asyncio.create_task(guard(scope(), receive, send))
        await asyncio.sleep(0)
        assert guard.waiting == 1 and guard.active["http"] == 1
        assert len(reads) == 1
        await guard(scope(), receive, send)
        assert replies[0]["status"] == 429 and len(reads) == 1
        release.set()
        await asyncio.gather(first, second)
        assert len(reads) == 2 and not guard.waiting and not guard.active["http"]

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_waiting_timeout_and_cancellation_release_reservations(cancel):
    async def run():
        started = asyncio.Event()

        async def app(scope, receive, send):
            started.set()
            await asyncio.Event().wait()

        guard = RequestLimits(app, requests=1, waiting_requests=1, request_wait=0.02)
        replies = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            replies.append(message)

        first = asyncio.create_task(guard(scope(), receive, send))
        await started.wait()
        second = asyncio.create_task(guard(scope(), receive, send))
        await asyncio.sleep(0)
        assert guard.waiting == 1
        if cancel:
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert not replies
        else:
            await second
            assert replies[0]["status"] == 429
        assert not guard.waiting and guard.active["http"] == 1
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not guard.active["http"]
        # The queue reservation and handler slot can both be reused.
        assert await guard.admit("http")
        assert guard.active["http"] == 1 and not guard.waiting

    asyncio.run(run())


def test_pilot_guard_precedes_fastapi_json_validation_and_does_not_echo_input(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MONAILABEL_DEMO_HTTP_LIMITS", "1")
    with TestClient(
        create_app(tmp_path, chat_provider=ScriptedChat()), client=("127.0.0.1", 1)
    ) as http:
        content = json.dumps({"username": "owner", "password": "PRIVATE-CONTENT" * 2000})
        rejected = http.post("/api/auth/setup", content=content)
        assert rejected.status_code == 413 and "PRIVATE-CONTENT" not in rejected.text
        assert http.get("/api/auth/status").json()["setup_required"]
        accepted = http.post(
            "/api/auth/setup", json={"username": "owner", "password": "test-password-1234"}
        )
        assert accepted.status_code == 201
        assert http.get("/api/projects").status_code == 200


def test_dispatch_failure_and_cancelled_upload_close_spools_and_release_capacity(monkeypatch):
    files = []
    original = tempfile.SpooledTemporaryFile

    def tracked(*args, **kwargs):
        spool = original(*args, **kwargs)
        files.append(spool)
        return spool

    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", tracked)

    async def run():
        async def app(scope, receive, send):
            raise ValueError("test dispatch failure")

        guard = RequestLimits(app, requests=1)

        async def receive():
            return {"type": "http.request", "body": b"x" * 1100000, "more_body": False}

        async def send(message):
            raise AssertionError("No response expected")

        with pytest.raises(ValueError, match="test dispatch failure"):
            await guard(scope(), receive, send)
        assert files[-1].closed and not guard.active["http"]

        receiving = asyncio.Event()

        async def pending():
            receiving.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(guard(scope(), pending, send))
        await receiving.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert files[-1].closed and not guard.active["http"]

    asyncio.run(run())
