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

"""Different conversations cannot collide in a hash bucket; active memory stays bounded."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import AssistantReply, AssistantRequest, User


def test_independent_conversations_with_identical_prefix_can_run_together(http, monkeypatch):
    service = http.app.state.services
    assistant = service.assistants
    user = service.store.list(User)[0]
    gate = threading.Barrier(2)

    def answer(*args):
        gate.wait(timeout=5)
        return AssistantReply(assistant="fixture", message="Done")

    monkeypatch.setattr(assistant, "_run", answer)
    requests = [AssistantRequest(message="hello", request_id="a" * 8 + str(i) * 24) for i in (1, 2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda request: assistant.run(None, request, user), requests))
    assert len(replies) == 2 and not assistant.active


def test_duplicate_conversation_rejected_and_active_set_released_after_failure(http, monkeypatch):
    service = http.app.state.services
    assistant = service.assistants
    user = service.store.list(User)[0]
    request = AssistantRequest(message="hello")

    def duplicate(*args):
        with pytest.raises(Conflict, match="processing"):
            assistant.run(None, request, user)
        raise RuntimeError("Failure")

    monkeypatch.setattr(assistant, "_run", duplicate)
    with pytest.raises(RuntimeError, match="Failure"):
        assistant.run(None, request, user)
    assert not assistant.active
    assistant.active.update(str(i) for i in range(64))
    with pytest.raises(DomainError, match="busy"):
        assistant.run(None, request, user)
    assert len(assistant.active) == 64
