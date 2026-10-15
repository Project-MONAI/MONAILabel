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

from concurrent.futures import ThreadPoolExecutor

import pytest
from chat_fixture import ScriptedChat
from fastapi.testclient import TestClient

from monailabel.server import request_throttle
from monailabel.server.app import create_app
from monailabel.server.demo_access import DemoBusy
from monailabel.server.request_throttle import RequestThrottle


def test_bucket_refills_without_extending_lockout_or_affecting_other_accounts(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(request_throttle, "monotonic", lambda: clock[0])
    guard = RequestThrottle(enabled=True)
    for _ in range(guard.BURST):
        guard.check("visitor")
    for _ in range(10):
        with pytest.raises(DemoBusy) as error:
            guard.check("visitor")
        assert error.value.code == "request_throttled" and error.value.retry_after == 1
    guard.check("another-visitor")
    clock[0] += 1
    for _ in range(guard.RATE):
        guard.check("visitor")
    with pytest.raises(DemoBusy):
        guard.check("visitor")
    clock[0] += guard.BURST / guard.RATE
    guard.check("new-visitor")
    assert set(guard.accounts) == {("new-visitor", False)}


def test_concurrent_requests_cannot_overdraw_or_grow_the_bucket_map(monkeypatch):
    monkeypatch.setattr(request_throttle, "monotonic", lambda: 100.0)
    guard = RequestThrottle(enabled=True)
    guard.BURST = 10
    guard.MAX_BUCKETS = 2

    def attempt(_):
        try:
            guard.check("visitor")
            return True
        except DemoBusy:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(100))) == 10
    guard.check("second")
    for index in range(100):
        with pytest.raises(DemoBusy):
            guard.check(f"new-{index}")
    assert len(guard.accounts) == 2
    disabled = RequestThrottle(enabled=False)
    for _ in range(100):
        disabled.check("visitor")
    assert not disabled.accounts


def test_http_rate_limit_is_shared_by_cookie_and_bearer_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_HTTP_LIMITS", "1")
    monkeypatch.setattr(request_throttle, "monotonic", lambda: 100.0)
    with TestClient(create_app(tmp_path, chat_provider=ScriptedChat())) as http:
        response = http.post(
            "/api/auth/setup", json={"username": "owner", "password": "disposable-test-password"}
        )
        assert response.status_code == 201
        services = http.app.state.services
        user = services.auth.authenticate(http.cookies.get("monailabel_session"))
        second_session = services.auth.issue(user)
        other = services.auth.create_user("another", "another-test-password")
        other_session = services.auth.issue(other)
        services.request_throttle.READ_BURST = 2
        assert http.get("/api/projects").status_code == 200
        assert (
            http.get(
                "/api/projects", headers={"Authorization": f"Bearer {second_session}"}
            ).status_code
            == 200
        )
        response = http.get("/api/projects")
        assert response.status_code == 429 and response.headers["Retry-After"] == "1"
        assert response.json()["code"] == "request_throttled"
        assert (
            http.get(
                "/api/projects", headers={"Authorization": f"Bearer {other_session}"}
            ).status_code
            == 200
        )
        before = services.request_throttle.accounts.copy()
        assert (
            http.get("/api/projects", headers={"Authorization": "Bearer invalid"}).status_code
            == 401
        )
        assert services.request_throttle.accounts == before


def test_viewer_read_bursts_do_not_consume_write_budget(monkeypatch):
    monkeypatch.setattr(request_throttle, "monotonic", lambda: 100.0)
    guard = RequestThrottle(enabled=True)
    for _ in range(guard.READ_BURST):
        guard.check("visitor", read=True)
    with pytest.raises(DemoBusy):
        guard.check("visitor", read=True)
    for _ in range(guard.BURST):
        guard.check("visitor")
    with pytest.raises(DemoBusy):
        guard.check("visitor")
