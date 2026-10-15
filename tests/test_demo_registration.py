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
from monailabel.core.models import Job, Membership, ModelRecord, Project, User
from monailabel.server.auth import Auth, LoginSession, Password
from monailabel.server.demo_access import DemoAccess, admission
from monailabel.server.desktops.models import DesktopSession


@pytest.fixture
def registration(http, monkeypatch):
    for key in (
        "MONAILABEL_DEMO_SELF_REGISTER",
        "MONAILABEL_DEMO_CACHE_ONLY",
        "MONAILABEL_DEMO_TRAINING_PREVIEW",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "2")
    http.cookies.clear()
    return http


def credentials(name="visitor-one"):
    return {"username": name, "password": "test-password-1234"}


def expire(service, identifier):
    with service.store.transaction() as session:
        lease = session.get(DemoAccess, identifier)
        session.update(
            lease.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        )


def test_registration_disabled_by_default_and_requires_all_guards(http, monkeypatch):
    http.cookies.clear()
    for key in (
        "MONAILABEL_DEMO_SELF_REGISTER",
        "MONAILABEL_DEMO_CACHE_ONLY",
        "MONAILABEL_DEMO_TRAINING_PREVIEW",
    ):
        assert http.get("/api/auth/status").json()["demo_registration"]["enabled"] is False
        assert http.post("/api/auth/register-demo", json=credentials()).status_code == 403
        monkeypatch.setenv(key, "1")
    assert http.get("/api/auth/status").json()["demo_registration"]["enabled"] is True


def test_new_visitor_owns_only_their_starter_project_and_has_fixed_deadline(
    registration, monkeypatch, tmp_path
):
    monkeypatch.setenv("MONAILABEL_MODELS_DIR", str(tmp_path / "missing-models"))
    http = registration
    service = http.app.state.services
    response = http.post("/api/auth/register-demo", json=credentials())
    assert response.status_code == 201, response.text
    assert response.headers["cache-control"] == "no-store"
    assert "httponly" in response.headers["set-cookie"].lower()
    user = response.json()
    assert user["is_admin"] is False
    assert http.get("/api/auth/me").json()["id"] == user["id"]
    projects = http.get("/api/projects").json()
    assert len(projects) == 1 and projects[0]["name"] == "My demo"
    project = projects[0]
    models = http.get(f"/api/projects/{project['id']}/models").json()
    assert models == []  # Host metadata is not proof of prepared visitor inference.
    assert {m.provider for m in service.store.list(ModelRecord, project["id"])} >= {
        "vista3d",
        "totalsegmentator-ct",
        "totalsegmentator-mr",
    }
    info = http.get("/api/auth/session").json()
    assert info["self_registered"] is True and info["initial_project_id"] == project["id"]
    lease = service.store.get(DemoAccess, user["id"])
    assert (lease.expires_at - lease.started_at).total_seconds() == 3600
    token = http.post("/api/auth/token").json()["token"]
    assert (
        service.store.get(LoginSession, hashlib.sha256(token.encode()).hexdigest()).expires_at
        == lease.expires_at
    )
    http.post("/api/auth/logout").raise_for_status()
    http.post("/api/auth/login", json=credentials()).raise_for_status()
    assert service.store.get(DemoAccess, user["id"]) == lease
    assert http.get("/api/auth/users").status_code == 403
    assert http.post("/api/auth/users", json=credentials("forbidden")).status_code == 403
    assert (
        http.post(
            f"/api/projects/{project['id']}/train", json={"snapshot_id": "missing"}
        ).status_code
        == 403
    )
    assert service.store.list(Job) == []
    expire(service, user["id"])
    assert http.get("/api/projects").status_code == 401
    assert http.post("/api/auth/login", json=credentials()).json()["code"] == "demo_expired"
    assert http.post("/api/auth/register-demo", json=credentials()).status_code == 409


def test_two_visitors_cannot_read_each_others_project_or_models(registration):
    http = registration
    http.post("/api/auth/register-demo", json=credentials()).raise_for_status()
    first = http.get("/api/projects").json()[0]
    http.cookies.clear()
    http.post("/api/auth/register-demo", json=credentials("visitor-two")).raise_for_status()
    second = http.get("/api/projects").json()
    assert len(second) == 1 and second[0]["id"] != first["id"]
    for route in ("", "/models", "/assets", "/members", "/jobs"):
        assert http.get(f"/api/projects/{first['id']}" + route).status_code == 403


def test_capacity_rejects_without_creating_records_and_survives_auth_restart(registration):
    http = registration
    service = http.app.state.services
    for name in ("visitor-one", "visitor-two"):
        http.post("/api/auth/register-demo", json=credentials(name)).raise_for_status()
        http.post("/api/auth/logout").raise_for_status()
    before = {
        kind: service.store.list(kind)
        for kind in (User, Password, Project, Membership, ModelRecord, DemoAccess, LoginSession)
    }
    response = http.post("/api/auth/register-demo", json=credentials("visitor-three"))
    assert response.status_code == 429 and response.json()["code"] == "demo_busy"
    assert 1 <= int(response.headers["retry-after"]) <= 3600
    assert http.get("/api/auth/status").json()["demo_registration"]["busy"] is True
    for kind, records in before.items():
        assert service.store.list(kind) == records
    with pytest.raises(DomainError) as error:
        Auth(service.store).register_demo(**credentials("visitor-four"))
    assert error.value.status == 429
    # Capacity never prevents an already-admitted visitor from signing back in.
    assert http.post("/api/auth/login", json=credentials()).status_code == 200


@pytest.mark.parametrize("resource", ["none", "job", "desktop"])
def test_expired_unfinished_resources_keep_admission_slot(registration, monkeypatch, resource):
    http = registration
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "1")
    user = http.post("/api/auth/register-demo", json=credentials()).json()
    service = http.app.state.services
    lease = service.store.get(DemoAccess, user["id"])
    expire(service, user["id"])
    with service.store.transaction() as session:
        if resource == "job":
            session.insert(Job(project_id=lease.initial_project_id, kind="fixture", request={}))
        elif resource == "desktop":
            session.insert(
                DesktopSession(
                    project_id=lease.initial_project_id,
                    user_id=user["id"],
                    asset_id="fixture",
                    viewer="slicer",
                    mode="annotation",
                    credential_id="fixture",
                )
            )
    status = admission(service.store)
    assert status["busy"] is (resource != "none")
    response = http.post("/api/auth/register-demo", json=credentials("visitor-two"))
    assert response.status_code == (201 if resource == "none" else 429)


def test_concurrent_registration_cannot_overbook(registration):
    store = registration.app.state.services.store

    # Separate Auth objects exercise the database transaction, not just one lock.
    def attempt(index):
        try:
            Auth(store).register_demo(**credentials(f"visitor-{index}"))
            return 201
        except DomainError as error:
            return error.status

    with ThreadPoolExecutor(max_workers=6) as executor:
        statuses = list(executor.map(attempt, range(6)))
    assert statuses.count(201) == 2 and statuses.count(429) == 4
    assert len(store.list(DemoAccess)) == 2
    assert len(store.list(Project)) == 2


def test_rate_limit_and_atomic_rollback(registration, monkeypatch):
    service = registration.app.state.services

    def fail(*args):
        raise RuntimeError("Fixture failure before commit")

    monkeypatch.setattr("monailabel.server.auth.create_project", fail)
    with pytest.raises(RuntimeError, match="Fixture failure"):
        service.auth.register_demo(**credentials())
    assert len(service.store.list(User)) == 1
    assert service.store.list(Project) == [] and service.store.list(DemoAccess) == []
    for _ in range(9):
        assert (
            registration.post("/api/auth/register-demo", json=credentials("bad name")).status_code
            == 422
        )
    response = registration.post("/api/auth/register-demo", json=credentials())
    assert response.status_code == 429 and response.headers["retry-after"] == "60"


def test_registration_cannot_elevate_or_leak_password(registration):
    response = registration.post(
        "/api/auth/register-demo", json={**credentials(), "is_admin": True}
    )
    assert response.status_code == 422
    assert credentials()["password"] not in response.text
    response = registration.post(
        "/api/auth/register-demo", json={**credentials(), "password": "short"}
    )
    assert response.status_code == 422 and "short" not in response.text
