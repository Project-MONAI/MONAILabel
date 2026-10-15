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
from contextlib import ExitStack

import pytest

from monailabel.core.errors import DomainError
from monailabel.server.login_throttle import LoginThrottle


def test_account_limit_is_case_insensitive_and_expires(monkeypatch):
    tick = [100.0]
    monkeypatch.setattr("monailabel.server.login_throttle.monotonic", lambda: tick[0])
    throttle = LoginThrottle()
    for index in range(10):
        with throttle.attempt(" Visitor ", f"source-{index}"):
            pass
    with pytest.raises(DomainError) as error, throttle.attempt("VISITOR", "new-source"):
        pytest.fail("A blocked attempt must not enter password verification")
    assert error.value.status == 429
    assert error.value.retry_after == 60
    tick[0] += 60
    with throttle.attempt("visitor", "new-source"):
        pass
    assert len(throttle._attempts) == 3


def test_global_limit_bounds_memory_with_rotating_accounts_and_sources(monkeypatch):
    monkeypatch.setattr("monailabel.server.login_throttle.monotonic", lambda: 100.0)
    throttle = LoginThrottle()
    for index in range(120):
        with throttle.attempt(f"user-{index}", f"source-{index}"):
            pass
    for index in range(1000):
        with pytest.raises(DomainError), throttle.attempt(f"new-{index}", f"new-{index}"):
            pytest.fail("Global limit must precede new bucket allocation")
    assert len(throttle._attempts) == 241


def test_parallel_attempts_cannot_overbook_account_window(monkeypatch):
    monkeypatch.setattr("monailabel.server.login_throttle.monotonic", lambda: 100.0)
    throttle = LoginThrottle()

    def attempt(index):
        try:
            with throttle.attempt("visitor", f"source-{index}"):
                return 200
        except DomainError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(attempt, range(40)))
    assert results.count(200) == 10
    assert results.count(429) == 30


def test_concurrent_hashing_is_bounded_and_slots_are_released():
    throttle = LoginThrottle()
    with ExitStack() as stack:
        for index in range(4):
            stack.enter_context(throttle.attempt(f"user-{index}", "source"))
        with pytest.raises(DomainError), throttle.attempt("fifth", "source"):
            pytest.fail("Password hashing must not queue when every slot is occupied")
    with pytest.raises(ValueError), throttle.attempt("failure", "source"):
        raise ValueError("fixture")
    with ExitStack() as stack:
        for index in range(4):
            stack.enter_context(throttle.attempt(f"next-{index}", "source"))


@pytest.mark.parametrize("username", ["owner", "unknown-user"])
def test_http_login_limits_before_hashing_without_leaking_credentials(http, monkeypatch, username):
    from monailabel.server import auth

    monkeypatch.setattr("monailabel.server.login_throttle.monotonic", lambda: 100.0)
    calls = []
    real_digest = auth.digest

    def recorded_digest(password, salt):
        calls.append(None)
        return real_digest(password, salt)

    monkeypatch.setattr(auth, "digest", recorded_digest)
    credentials = {"username": username, "password": "wrong-secret-not-logged"}
    for _ in range(10):
        assert http.post("/api/auth/login", json=credentials).status_code == 401
    response = http.post("/api/auth/login", json=credentials)
    assert response.status_code == 429
    assert response.json()["code"] == "login_throttled"
    assert response.headers["retry-after"] == "60"
    assert response.headers["cache-control"] == "no-store"
    assert credentials["password"] not in response.text
    assert len(calls) == 10


def test_forwarded_headers_cannot_bypass_source_limit(http, monkeypatch):
    monkeypatch.setattr("monailabel.server.login_throttle.monotonic", lambda: 100.0)
    for index in range(31):
        response = http.post(
            "/api/auth/login",
            json={"username": f"unknown-{index}", "password": "wrong-password"},
            headers={"X-Forwarded-For": f"192.0.2.{index + 1}"},
        )
        assert response.status_code == (401 if index < 30 else 429)


def test_successful_logins_are_also_bounded(http, monkeypatch):
    monkeypatch.setattr("monailabel.server.login_throttle.monotonic", lambda: 100.0)
    credentials = {"username": "owner", "password": "test-password-1234"}
    for _ in range(10):
        assert http.post("/api/auth/login", json=credentials).status_code == 200
    assert http.post("/api/auth/login", json=credentials).status_code == 429


def test_completed_setup_does_not_hash_another_password(http, monkeypatch):
    def unexpected_hash(*args):
        pytest.fail("An initialized server must reject setup before hashing")

    monkeypatch.setattr("monailabel.server.auth.digest", unexpected_hash)
    response = http.post(
        "/api/auth/setup", json={"username": "second", "password": "test-password-1234"}
    )
    assert response.status_code == 409
