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

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from chat_fixture import ScriptedChat
from fastapi.testclient import TestClient

from monailabel.core.chat import Conversation
from monailabel.core.errors import DomainError
from monailabel.core.models import (
    Job,
    JobStatus,
    Membership,
    ModelRecord,
    Project,
    ProjectCreate,
    User,
)
from monailabel.server.app import create_app
from monailabel.server.demo_access import DemoAccess, DemoVisit
from monailabel.server.demo_compute import isolated
from monailabel.server.demo_context import acting_as
from monailabel.server.demo_lifecycle import DemoFeedback
from monailabel.server.desktops.models import DesktopSession
from monailabel.server.jobs import Outcome
from monailabel.server.video.models import VideoEditor


@pytest.fixture
def demo_http(tmp_path, monkeypatch):
    for key in ("LIFECYCLE", "SELF_REGISTER", "CACHE_ONLY", "TRAINING_PREVIEW", "AUDIT"):
        monkeypatch.setenv("MONAILABEL_DEMO_" + key, "1")
    with TestClient(create_app(tmp_path, chat_provider=ScriptedChat())) as http:
        http.post(
            "/api/auth/setup", json={"username": "owner", "password": "test-password-1234"}
        ).raise_for_status()
        yield http


def register(http, name="visitor"):
    response = http.post(
        "/api/auth/register-demo", json={"username": name, "password": "test-password-1234"}
    )
    response.raise_for_status()
    return response.json()["id"]


def admin(http):
    http.post(
        "/api/auth/login", json={"username": "owner", "password": "test-password-1234"}
    ).raise_for_status()


def expire(http, identifier):
    with http.app.state.services.store.transaction() as session:
        lease = session.get(DemoAccess, identifier)
        session.update(
            lease.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        )


def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate()


def reset_body(http):
    preview = http.get("/api/demo-access/reset-preview")
    preview.raise_for_status()
    return {
        "confirmation": "START FRESH DEMO",
        "confirmation_token": preview.json()["confirmation_token"],
    }


@pytest.mark.parametrize("minutes", ["60", "120"])
def test_absolute_expiry_without_browser_cancels_owned_jobs_and_viewers(
    demo_http, monkeypatch, minutes
):
    http = demo_http
    monkeypatch.setenv("MONAILABEL_DEMO_MINUTES", minutes)
    owner = register(http)
    service = http.app.state.services
    lease = service.store.get(DemoAccess, owner)
    assert (lease.expires_at - lease.started_at).total_seconds() == int(minutes) * 60
    project = lease.initial_project_id
    token = http.post("/api/auth/token").json()["token"]
    from monailabel.server.auth import LoginSession

    credential = next(s for s in service.store.list(LoginSession) if s.user_id == owner)
    desktop = DesktopSession(
        project_id=project,
        user_id=owner,
        asset_id="test",
        viewer="qupath",
        mode="annotation",
        credential_id=credential.id,
    )
    editor = VideoEditor(
        project_id=project,
        user_id=owner,
        asset_id="test",
        base_revision=0,
        mode="annotation",
        server_url="http://127.0.0.1:9999",
        task_id=1,
        ready=True,
    )
    with service.store.transaction() as session:
        session.insert(desktop)
        session.insert(editor)
    stopped = []
    monkeypatch.setattr(
        service.desktops.runtime, "stop", lambda identifier: stopped.append(identifier)
    )
    started = threading.Event()

    def work(context):
        started.set()
        while True:
            context.progress(0.2)
            time.sleep(0.02)

    with acting_as(owner):
        job = service.jobs.submit("test", project, {}, work)
    assert started.wait(3)
    other = register(http, "other-visitor")
    other_deadline = service.store.get(DemoAccess, other).expires_at
    expire(http, owner)
    until(lambda: not service.jobs.executing())
    until(lambda: service.store.get(DemoAccess, owner).stopped_at is not None)
    assert service.store.get(Job, job.id).status == JobStatus.CANCELLED
    assert stopped == [desktop.id]
    assert service.store.get(DesktopSession, desktop.id).ended
    assert service.store.get(VideoEditor, editor.id).ended
    assert service.store.get(DemoAccess, other).expires_at == other_deadline
    assert (
        http.get("/api/projects", headers={"Authorization": "Bearer " + token}).status_code == 401
    )
    assert http.get("/api/auth/session").status_code == 200


def test_reset_requires_admin_fresh_confirmation_and_preserves_feedback(demo_http, tmp_path):
    http = demo_http
    service = http.app.state.services
    admin_project = http.post("/api/projects", json={"name": "Keep administrator work"}).json()
    ordinary = User(username="ordinary-account")
    with service.store.transaction() as session:
        session.insert(ordinary)
    owner = register(http)
    project = service.store.get(DemoAccess, owner).initial_project_id
    old_token = http.post("/api/auth/token").json()["token"]
    conversation = Conversation(user_id=owner, project_id=project)
    with service.store.transaction() as session:
        session.insert(conversation)
        shared = Project(name="Keep mixed membership")
        session.insert(shared)
        session.insert(Membership(project_id=shared.id, user_id=owner, roles=["annotator"]))
        session.insert(Membership(project_id=shared.id, user_id=ordinary.id, roles=["annotator"]))
    cache = tmp_path / ".cache" / "prepared"
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(b"keep shared model and dataset")
    assert http.get("/api/demo-access/reset-preview").status_code == 403
    assert (
        http.post(
            "/api/demo-access/reset",
            json={"confirmation": "START FRESH DEMO", "confirmation_token": "0" * 64},
        ).status_code
        == 403
    )
    admin(http)
    body = reset_body(http)
    assert (
        http.post("/api/demo-access/reset", json=body | {"confirmation": "wrong"}).status_code
        == 422
    )
    register(http, "second-visitor")
    admin(http)
    assert http.post("/api/demo-access/reset", json=body).status_code == 409
    response = http.post("/api/demo-access/reset", json=reset_body(http))
    assert response.status_code == 200, response.text
    assert response.json()["removed_visitors"] == 2
    assert http.get("/api/projects/" + admin_project["id"]).status_code == 200
    assert {u.username for u in service.store.list(User)} == {"owner", "ordinary-account"}
    assert service.store.get(Project, shared.id).name == "Keep mixed membership"
    assert service.store.list(DemoAccess) == []
    assert service.store.list(DemoFeedback)[0].source_id == conversation.id
    assert service.store.get(DemoVisit, owner).reset_at is not None
    assert cache.read_bytes() == b"keep shared model and dataset"
    replacement = register(http)
    assert replacement != owner
    assert (
        service.store.get(DemoVisit, replacement).round_id
        != service.store.get(DemoVisit, owner).round_id
    )
    assert (
        http.get("/api/projects", headers={"Authorization": "Bearer " + old_token}).status_code
        == 401
    )
    assert http.get("/api/demo-access/reset-preview").status_code == 403


def test_restart_preserves_visitors_and_their_absolute_deadline(tmp_path, monkeypatch):
    for key in ("LIFECYCLE", "SELF_REGISTER", "CACHE_ONLY", "TRAINING_PREVIEW"):
        monkeypatch.setenv("MONAILABEL_DEMO_" + key, "1")
    with TestClient(create_app(tmp_path, chat_provider=ScriptedChat())) as http:
        http.post(
            "/api/auth/setup", json={"username": "owner", "password": "test-password-1234"}
        ).raise_for_status()
        identifier = register(http)
        deadline = http.app.state.services.store.get(DemoAccess, identifier).expires_at
    # The original server is closed before opening the same disposable workspace.
    with TestClient(create_app(tmp_path, chat_provider=ScriptedChat())) as http:
        assert http.app.state.services.store.get(User, identifier).username == "visitor"
        assert http.app.state.services.store.get(DemoAccess, identifier).expires_at == deadline
        http.post(
            "/api/auth/login", json={"username": "visitor", "password": "test-password-1234"}
        ).raise_for_status()


def test_reset_waits_for_cancelled_work_to_really_finish(demo_http):
    http = demo_http
    owner = register(http)
    service = http.app.state.services
    project = service.store.get(DemoAccess, owner).initial_project_id
    started, release = threading.Event(), threading.Event()

    def work(context):
        started.set()
        release.wait(5)
        return Outcome({})

    with acting_as(owner):
        job = service.jobs.submit("test", project, {}, work)
    assert started.wait(3)
    service.jobs.cancel(job.id)
    admin(http)
    try:
        assert http.post("/api/demo-access/reset", json=reset_body(http)).status_code == 409
    finally:
        release.set()
    until(lambda: not service.jobs.executing())
    assert http.post("/api/demo-access/reset", json=reset_body(http)).status_code == 200


def test_stale_request_cannot_publish_after_expiry_or_reset(demo_http):
    http = demo_http
    owner = register(http)
    service = http.app.state.services
    expire(http, owner)
    with acting_as(owner), pytest.raises(DomainError):
        service.datasets.create(ProjectCreate(name="Too late"))
    admin(http)
    http.post("/api/demo-access/reset", json=reset_body(http)).raise_for_status()
    with acting_as(owner), pytest.raises(DomainError), service.store.transaction() as session:
        session.insert(Project(name="Still too late"))


def test_http_jobs_retain_visitor_identity(demo_http):
    http = demo_http
    owner = register(http)
    project = http.app.state.services.store.get(DemoAccess, owner).initial_project_id
    response = http.post(f"/api/projects/{project}/select", json={"strategy": "random", "limit": 1})
    response.raise_for_status()
    assert response.json()["user_id"] == owner


def test_failed_desktop_stop_holds_slot_until_retry(demo_http, monkeypatch):
    from monailabel.server.auth import LoginSession

    http = demo_http
    owner = register(http)
    service = http.app.state.services
    lease = service.store.get(DemoAccess, owner)
    token = next(t for t in service.store.list(LoginSession) if t.user_id == owner)
    desktop = DesktopSession(
        project_id=lease.initial_project_id,
        user_id=owner,
        asset_id="test",
        viewer="slicer",
        mode="annotation",
        credential_id=token.id,
    )
    with service.store.transaction() as session:
        session.insert(desktop)

    def fail(identifier):
        raise RuntimeError("Test-only unavailable Docker")

    monkeypatch.setattr(service.desktops.runtime, "stop", fail)
    expire(http, owner)
    service.demo_lifecycle.sweep()
    assert service.store.get(DemoAccess, owner).stopped_at is None
    assert not service.store.get(DesktopSession, desktop.id).ended
    monkeypatch.setattr(service.desktops.runtime, "stop", lambda identifier: None)
    service.demo_lifecycle.sweep()
    assert service.store.get(DemoAccess, owner).stopped_at


def test_running_cvat_decode_keeps_reservation_and_blocks_reset(demo_http):
    from types import SimpleNamespace

    http = demo_http
    owner = register(http)
    service = http.app.state.services
    lease = service.store.get(DemoAccess, owner)
    editor = VideoEditor(
        project_id=lease.initial_project_id,
        user_id=owner,
        asset_id="test",
        base_revision=0,
        mode="annotation",
        server_url="http://cvat.test",
        task_id=1,
        preparation_request_id="test-request",
    )
    with service.store.transaction() as session:
        session.insert(editor)
    done = False
    service.video_editors.client = SimpleNamespace(
        url=editor.server_url, cancel_preparation=lambda identifier: done, close=lambda: None
    )
    expire(http, owner)
    service.demo_lifecycle.sweep()
    assert service.store.get(VideoEditor, editor.id).ended
    assert service.store.get(DemoAccess, owner).stopped_at is None
    admin(http)
    assert http.post("/api/demo-access/reset", json=reset_body(http)).status_code == 409
    done = True
    service.demo_lifecycle.sweep()
    assert service.store.get(DemoAccess, owner).stopped_at
    assert http.post("/api/demo-access/reset", json=reset_body(http)).status_code == 200


@pytest.mark.parametrize(
    "state,cancel_status,expected",
    [
        ("queued", 200, True),
        ("started", 400, False),
        ("finished", None, True),
        ("failed", None, True),
    ],
)
def test_cvat_cancellation_does_not_claim_started_import_stopped(state, cancel_status, expected):
    import httpx

    from monailabel.viewers.cvat import CvatClient

    client = CvatClient("http://cvat.test", "test-token")
    client.http.close()
    calls = []

    def handler(request):
        calls.append(request.method)
        return (
            httpx.Response(200, json={"status": state})
            if request.method == "GET"
            else httpx.Response(cancel_status)
        )

    client.http = httpx.Client(base_url=client.url, transport=httpx.MockTransport(handler))
    try:
        assert client.cancel_preparation("test-request") is expected
        assert calls == (["GET"] if cancel_status is None else ["GET", "POST"])
    finally:
        client.close()


@pytest.mark.parametrize("gpu", [None, "0", "1"])
def test_real_isolated_prediction_returns_valid_result(demo_http, monkeypatch, gpu):
    import monailabel.server.demo_compute as compute

    original = compute.subprocess.Popen
    environments = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    if gpu is not None:
        monkeypatch.setenv("MONAILABEL_DEMO_GPU_DEVICES", gpu)
    else:
        monkeypatch.delenv("MONAILABEL_DEMO_GPU_DEVICES", raising=False)

    def start(*args, **kwargs):
        environments.append(kwargs["env"])
        return original(*args, **kwargs)

    monkeypatch.setattr(compute.subprocess, "Popen", start)
    http = demo_http
    owner = register(http)
    service = http.app.state.services
    project = service.store.get(Project, service.store.get(DemoAccess, owner).initial_project_id)
    model = ModelRecord(
        project_id=project.id,
        name="Synthetic threshold",
        provider="threshold",
        label_ids=[0, 1],
        config={"thresholds": [0.5]},
    )

    def work(context):
        result = service.models.predict(project, model, np.zeros((12, 10, 1), np.float32))
        assert result.shape == (12, 10)
        return Outcome({"checked": True})

    with acting_as(owner):
        job = service.jobs.submit("annotate", project.id, {}, work)
    until(
        lambda: service.store.get(Job, job.id).status in {JobStatus.SUCCEEDED, JobStatus.FAILED},
        timeout=30,
    )
    assert service.store.get(Job, job.id).result == {"checked": True}
    assert environments
    assert all(env["MONAILABEL_PREPARED_MODELS_ONLY"] == "1" for env in environments)
    assert all(env["CUDA_VISIBLE_DEVICES"] == (gpu or "0,1") for env in environments)
    assert compute.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.skipif(sys.platform != "linux", reason="Private Linux demo process groups")
def test_expiry_kills_uncooperative_inference_and_child_process(demo_http, monkeypatch, tmp_path):
    import monailabel.server.demo_compute as compute

    http = demo_http
    owner = register(http)
    service = http.app.state.services
    project = service.store.get(DemoAccess, owner).initial_project_id
    original = subprocess.Popen
    processes = []
    marker = tmp_path / "child-pid"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )

    def start(command, **kwargs):
        process = original([sys.executable, "-c", script, str(marker)], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(compute.subprocess, "Popen", start)

    def work(context):
        isolated("predict")
        return Outcome({"must_not_publish": True})

    with acting_as(owner):
        job = service.jobs.submit("annotate", project, {}, work)
    until(marker.exists)
    child = int(marker.read_text())
    try:
        expire(http, owner)
        until(lambda: not service.jobs.executing())
        assert processes[0].poll() is not None
        # Linux may retain a killed orphan as a zombie briefly; it is not executing.
        state = Path(f"/proc/{child}/stat")
        until(lambda: not state.exists() or state.read_text().split()[2] == "Z")
        assert service.store.get(Job, job.id).status == JobStatus.CANCELLED
        assert service.store.get(Job, job.id).result == {}
        from monailabel.server.temporary_storage import TemporaryReservations

        reservations = TemporaryReservations(service.store.path.parent / "temporary-spools")
        with reservations.connection() as connection:
            assert connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
        assert not any(path.is_dir() for path in reservations.root.iterdir())
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
