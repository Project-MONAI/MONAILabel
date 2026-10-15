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

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from monailabel.core.errors import DomainError
from monailabel.server.auth import Auth, LoginSession
from monailabel.server.demo_access import DemoAccess
from monailabel.server.desktops.models import DesktopSession


def visitor(http, minutes=60):
    result = http.post(
        "/api/auth/demo-users",
        json={"username": "visitor", "password": "test-password-1234", "minutes": minutes},
    )
    assert result.status_code == 201, result.text
    return result.json()


def expire(service, user_id):
    with service.store.transaction() as session:
        lease = session.get(DemoAccess, user_id)
        session.update(
            lease.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        )


@pytest.mark.parametrize("minutes", [60, 120])
def test_deadline_starts_once_and_survives_tokens_logout_and_auth_restart(http, minutes):
    service = http.app.state.services
    user = visitor(http, minutes)
    assert service.store.get(DemoAccess, user["id"]).expires_at is None
    credentials = {"username": "visitor", "password": "test-password-1234"}
    assert (
        http.post("/api/auth/login", json={**credentials, "password": "wrong"}).status_code == 401
    )
    assert service.store.get(DemoAccess, user["id"]).expires_at is None
    assert http.post("/api/auth/login", json=credentials).status_code == 200
    lease = service.store.get(DemoAccess, user["id"])
    assert (lease.expires_at - lease.started_at).total_seconds() == minutes * 60
    token = http.post("/api/auth/token").json()["token"]
    session = service.store.get(LoginSession, hashlib.sha256(token.encode()).hexdigest())
    assert session.expires_at == lease.expires_at
    assert http.post("/api/auth/logout").status_code == 200
    restarted = Auth(service.store)
    restarted.login(**credentials)
    assert service.store.get(DemoAccess, user["id"]) == lease
    expire(service, user["id"])
    with pytest.raises(DomainError, match="expired"):
        restarted.login(**credentials)
    assert http.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert http.post("/api/auth/login", json=credentials).status_code == 401
    assert (
        http.post(
            "/api/auth/login", json={"username": "owner", "password": "test-password-1234"}
        ).status_code
        == 200
    )
    assert http.get("/api/auth/session").json()["demo"] is False
    assert http.post("/api/auth/demo-users", json=credentials).status_code == 409


def test_visitor_cannot_mint_or_renew_own_invitation(http):
    visitor(http)
    credentials = {"username": "visitor", "password": "test-password-1234"}
    http.post("/api/auth/login", json=credentials).raise_for_status()
    assert http.post("/api/auth/demo-users", json=credentials).status_code == 403
    assert http.get("/api/demo-access/audit-status").status_code == 403
    assert http.get("/api/demo-access/events").status_code == 405


def test_parallel_logins_share_one_deadline(http):
    visitor(http)
    auth = http.app.state.services.auth
    with ThreadPoolExecutor(max_workers=4) as executor:
        tokens = list(
            executor.map(lambda _: auth.login("visitor", "test-password-1234")[1], range(4))
        )
    deadlines = {
        auth.store.get(LoginSession, hashlib.sha256(token.encode()).hexdigest()).expires_at
        for token in tokens
    }
    assert len(deadlines) == 1


def test_desktop_credential_renewal_cannot_extend_visitor_deadline(http):
    user = visitor(http)
    service = http.app.state.services
    _, token = service.auth.login("visitor", "test-password-1234")
    desktop = DesktopSession(
        project_id="fixture",
        asset_id="fixture",
        user_id=user["id"],
        viewer="slicer",
        mode="annotation",
        credential_id=hashlib.sha256(token.encode()).hexdigest(),
    )
    with service.store.transaction() as session:
        session.insert(desktop)
    service.desktops.renew(desktop)
    lease = service.store.get(DemoAccess, user["id"])
    assert service.store.get(LoginSession, desktop.credential_id).expires_at == lease.expires_at
    expire(service, user["id"])
    with pytest.raises(DomainError, match="expired"):
        service.desktops.renew(desktop)


def test_invited_admission_still_enforces_capacity_when_signup_is_disabled(http, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_SELF_REGISTER", "0")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "1")
    auth = http.app.state.services.auth
    first = auth.create_user("first", "test-password-1234", demo_minutes=60)
    second = auth.create_user("second", "test-password-1234", demo_minutes=60)
    auth.login("first", "test-password-1234")
    before = auth.store.list(LoginSession)
    with pytest.raises(DomainError) as error:
        Auth(auth.store).login("second", "test-password-1234")
    assert error.value.status == 429 and error.value.code == "demo_busy"
    assert auth.store.get(DemoAccess, second.id).expires_at is None
    assert auth.store.list(LoginSession) == before
    assert http.get("/api/auth/status").json()["demo_registration"]["enabled"] is False
    assert (
        http.post(
            "/api/auth/register-demo",
            json={"username": "uninvited", "password": "test-password-1234"},
        ).status_code
        == 403
    )
    # Reconnecting an admitted visitor or the administrator does not need a new slot.
    auth.login("first", "test-password-1234")
    auth.login("owner", "test-password-1234")
    expire(http.app.state.services, first.id)
    assert auth.login("second", "test-password-1234")[0].id == second.id


def test_concurrent_invited_admission_cannot_overbook_without_signup(http, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_SELF_REGISTER", "0")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "2")
    store = http.app.state.services.store
    for index in range(6):
        http.app.state.services.auth.create_user(
            f"invited-{index}", "test-password-1234", demo_minutes=60
        )

    def attempt(index):
        try:
            Auth(store).login(f"invited-{index}", "test-password-1234")
            return 200
        except DomainError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(attempt, range(6)))
    assert results.count(200) == 2 and results.count(429) == 4
    assert sum(lease.started_at is not None for lease in store.list(DemoAccess)) == 2
