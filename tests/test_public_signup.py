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

from datetime import UTC, datetime, timedelta

import pytest

from monailabel.core.models import Job, JobStatus, Membership, Project, Role, User
from monailabel.server.auth import Auth
from monailabel.server.demo_access import DemoAccess
from monailabel.server.demo_context import actor


def credentials(name="public-visitor"):
    return {"username": name, "password": "test-password-1234"}


@pytest.fixture
def public_signup(http, monkeypatch):
    for key in (
        "MONAILABEL_DEMO_SELF_REGISTER",
        "MONAILABEL_DEMO_PUBLIC_REGISTER",
        "MONAILABEL_DEMO_CACHE_ONLY",
        "MONAILABEL_DEMO_TRAINING_PREVIEW",
        "MONAILABEL_DEMO_HTTP_LIMITS",
        "MONAILABEL_DEMO_LIFECYCLE",
    ):
        monkeypatch.setenv(key, "1")
    http.cookies.clear()
    return http


def test_signup_is_separate_from_admission_and_uses_least_privilege(public_signup):
    http = public_signup
    service = http.app.state.services
    response = http.post("/api/auth/signup", json=credentials())
    assert response.status_code == 201, response.text
    assert "set-cookie" not in response.headers
    user = response.json()
    assert not user["is_admin"]
    assert service.store.list(Project) == []
    lease = service.store.get(DemoAccess, user["id"])
    assert lease.restricted_roles and lease.expires_at is None and lease.started_at is None
    assert http.get("/api/projects").status_code == 401
    assert http.post("/api/auth/login", json=credentials()).status_code == 200
    projects = http.get("/api/projects").json()
    assert len(projects) == 1
    project = projects[0]["id"]
    assert set(http.get(f"/api/projects/{project}/permissions").json()["roles"]) == {
        "annotator",
        "reviewer",
    }
    for endpoint, body in (
        ("/api/projects", {"name": "privilege escalation"}),
        (f"/api/projects/{project}/members", {"user_id": user["id"], "roles": ["manager"]}),
    ):
        response = http.request(
            "PUT" if endpoint.endswith("members") else "POST", endpoint, json=body
        )
        assert response.status_code == 403, response.text
    assert http.get("/api/auth/accounts").status_code == 403
    assert http.post("/api/demo").status_code == 403
    first = service.store.get(DemoAccess, user["id"])
    assert (first.expires_at - first.started_at).total_seconds() == 3600
    assert http.post("/api/auth/login", json=credentials()).status_code == 200
    assert service.store.get(DemoAccess, user["id"]) == first


def test_approved_accounts_wait_without_consuming_time(public_signup, monkeypatch):
    http = public_signup
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "1")
    for name in ("first", "waiting"):
        assert http.post("/api/auth/signup", json=credentials(name)).status_code == 201
    assert http.post("/api/auth/login", json=credentials("first")).status_code == 200
    http.cookies.clear()
    assert http.post("/api/auth/login", json=credentials("waiting")).status_code == 429
    store = http.app.state.services.store
    waiting = next(u for u in store.list(User) if u.username == "waiting")
    lease = store.get(DemoAccess, waiting.id)
    assert (
        lease.started_at is None and lease.expires_at is None and lease.initial_project_id is None
    )
    assert len(store.list(Project)) == 1
    # Expiry plus confirmed resource cleanup releases the slot, including after Auth restart.
    with store.transaction() as session:
        first = next(d for d in session.list(DemoAccess) if d.id != waiting.id)
        session.update(
            first.model_copy(
                update={
                    "expires_at": datetime.now(UTC) - timedelta(seconds=1),
                    "stopped_at": datetime.now(UTC),
                }
            )
        )
    user, _ = Auth(store).login(**credentials("waiting"))
    assert user.id == waiting.id
    assert store.get(DemoAccess, waiting.id).started_at is not None


def test_admin_directory_includes_unassigned_and_waiting_accounts(public_signup):
    http = public_signup
    response = http.post("/api/auth/signup", json=credentials())
    assert response.status_code == 201
    assert http.get("/api/auth/accounts").status_code == 401
    assert http.post("/api/auth/login", json=credentials("owner")).status_code == 200
    response = http.get("/api/auth/accounts")
    assert response.status_code == 200
    accounts = {u["username"]: u for u in response.json()}
    assert accounts["owner"]["kind"] == "Administrator"
    assert accounts["public-visitor"]["status"] == "Approved — not started"
    assert accounts["public-visitor"]["access"] == "Annotator + Reviewer"
    assert credentials()["password"] not in response.text
    assert "digest" not in response.text and "salt" not in response.text


def test_public_requires_explicit_guards_and_rejects_role_injection(public_signup, monkeypatch):
    http = public_signup
    assert (
        http.post("/api/auth/signup", json={**credentials(), "is_admin": True}).status_code == 422
    )
    for guard in (
        "MONAILABEL_DEMO_PUBLIC_REGISTER",
        "MONAILABEL_DEMO_HTTP_LIMITS",
        "MONAILABEL_DEMO_LIFECYCLE",
    ):
        monkeypatch.setenv(guard, "0")
        assert http.post("/api/auth/signup", json=credentials()).status_code == 403
        monkeypatch.setenv(guard, "1")


def test_restricted_visitor_cannot_gain_manager_even_if_membership_is_changed(public_signup):
    http = public_signup
    user = http.post("/api/auth/signup", json=credentials()).json()
    http.post("/api/auth/login", json=credentials()).raise_for_status()
    service = http.app.state.services
    with service.store.transaction() as session:
        member = next(m for m in session.list(Membership) if m.user_id == user["id"])
        session.update(
            member.model_copy(update={"roles": [Role.MANAGER, Role.ANNOTATOR, Role.REVIEWER]})
        )
    roles = service.auth.roles(service.store.get(User, user["id"]), member.project_id)
    assert roles == {Role.ANNOTATOR, Role.REVIEWER}
    assert (
        http.request(
            "DELETE", f"/api/projects/{member.project_id}", json={"confirmation_name": "My demo"}
        ).status_code
        == 403
    )


def test_cached_starter_is_bound_to_visitor_and_submitted_once(public_signup, monkeypatch):
    http = public_signup
    service = http.app.state.services
    calls = []

    def start(project_id, request, user_id):
        assert actor.get() == user_id
        assert request.template_id == "Task09_Spleen" and request.limit == 1
        assert request.include_masks is False
        calls.append((project_id, user_id))
        job = Job(
            project_id=project_id,
            user_id=user_id,
            kind="dataset_import",
            request={},
            status=JobStatus.QUEUED,
        )
        with service.store.transaction() as session:
            session.insert(job)
        return job

    monkeypatch.setattr(service.dataset_templates, "start", start)
    http.post("/api/auth/signup", json=credentials()).raise_for_status()
    http.post("/api/auth/login", json=credentials()).raise_for_status()
    http.post("/api/demo-access/starter").raise_for_status()
    assert len(calls) == 1


def test_starter_retries_empty_results_with_a_finite_limit(public_signup, monkeypatch):
    from monailabel.core.errors import DomainError

    http = public_signup
    service = http.app.state.services
    user_id = http.post("/api/auth/signup", json=credentials()).json()["id"]
    http.post("/api/auth/login", json=credentials()).raise_for_status()
    user = service.store.get(User, user_id)
    project_id = service.store.get(DemoAccess, user_id).initial_project_id
    calls = []

    def start(*args):
        calls.append(args)
        job = Job(
            project_id=project_id, kind="dataset_import", status=JobStatus.SUCCEEDED, request={}
        )
        with service.store.transaction() as session:
            session.insert(job)
        return job

    monkeypatch.setattr(service.dataset_templates, "start", start)
    for _ in range(3):
        service.demo_onboarding.start(user)
    with pytest.raises(DomainError, match="host must check"):
        service.demo_onboarding.start(user)
    assert len(calls) == 3


def test_restricted_project_creation_is_denied_through_service(public_signup):
    from monailabel.core.errors import DomainError
    from monailabel.core.models import ProjectCreate

    http = public_signup
    service = http.app.state.services
    user_id = http.post("/api/auth/signup", json=credentials()).json()["id"]
    token = actor.set(user_id)
    try:
        with pytest.raises(DomainError, match="Annotator and Reviewer"):
            service.datasets.create(ProjectCreate(name="Tool must not elevate access"))
    finally:
        actor.reset(token)
