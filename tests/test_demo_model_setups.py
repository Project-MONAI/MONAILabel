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

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import (
    Job,
    Learner,
    ModelRecord,
    Project,
    Role,
    Snapshot,
    StartTraining,
    TrainRequest,
)
from monailabel.server.demo_access import DemoAccess
from monailabel.server.demo_context import acting_as


@pytest.fixture
def setup_visitor(http, monkeypatch):
    for flag in (
        "SELF_REGISTER",
        "PUBLIC_REGISTER",
        "CACHE_ONLY",
        "TRAINING_PREVIEW",
        "HTTP_LIMITS",
        "LIFECYCLE",
    ):
        monkeypatch.setenv("MONAILABEL_DEMO_" + flag, "1")
    service = http.app.state.services
    monkeypatch.setattr(service.demo_onboarding, "start", lambda *args, **kwargs: None)
    credentials = {"username": "setup-visitor", "password": "test-password-1234"}
    http.cookies.clear()
    user = http.post("/api/auth/signup", json=credentials).json()
    http.post("/api/auth/login", json=credentials).raise_for_status()
    project = http.get("/api/projects").json()[0]
    return service, project["id"], user["id"]


@pytest.mark.parametrize(
    "recipe", ["monai-unet", "vista3d", "totalsegmentator-ct", "totalsegmentator-mr"]
)
def test_visitor_creates_setup_but_cannot_train(http, setup_visitor, monkeypatch, recipe):
    service, pid, uid = setup_visitor
    before = service.store.list(ModelRecord)
    body = {"name": recipe + "-Spleen", "recipe": recipe, "label_ids": [0, 1]}
    if recipe != "monai-unet":
        body["initial_model_id"] = next(m.id for m in before if m.preset == recipe)
    response = http.post(f"/api/projects/{pid}/learners", json=body)
    assert response.status_code == 201, response.text
    learner = response.json()
    assert service.store.list(ModelRecord) == before
    assert not service.store.list(Job) and not service.store.list(Snapshot)
    assert set(http.get(f"/api/projects/{pid}/permissions").json()["roles"]) == {
        "annotator",
        "reviewer",
    }
    # The restriction persists even if the global training-preview switch is off.
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_PREVIEW", "0")
    assert (
        http.post(f"/api/projects/{pid}/learners/{learner['id']}/train", json={}).status_code == 403
    )
    with acting_as(uid):
        with pytest.raises(DomainError, match="Live training"):
            service.learning.start(pid, learner["id"], StartTraining())
        with pytest.raises(DomainError, match="Live training"):
            service.learning.train(pid, TrainRequest(snapshot_id="not-created"))
    assert not service.store.list(Job) and not service.store.list(Snapshot)


def test_chat_setup_and_start_now_boundary(http, setup_visitor):
    service, pid, _ = setup_visitor

    def prompt(start_now):
        service.assistants.provider.queue = [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="setup",
                        name="create_learner",
                        arguments={
                            "recipe": "vista3d",
                            "name": "VISTA3D-Spleen",
                            "targets": ["spleen"],
                            "start_now": start_now,
                        },
                    )
                ],
            )
        ]
        return http.post(
            f"/api/projects/{pid}/assistant",
            json={"message": 'Based on VISTA3D, create model "VISTA3D-Spleen" for spleen.'},
        )

    response = prompt(False)
    assert response.status_code == 200, response.text
    assert "setup only" in response.json()["message"]
    assert response.json()["job_id"] is None
    learners = service.store.list(Learner)
    assert len(learners) == 1 and learners[0].name == "VISTA3D-Spleen"
    prompt(True)
    assert service.store.list(Learner) == learners
    assert not service.store.list(Job) and not service.store.list(Snapshot)


def test_visitor_rejects_arbitrary_settings_and_expired_creation(http, setup_visitor):
    service, pid, uid = setup_visitor
    prefix = f"/api/projects/{pid}"
    for body in (
        {"recipe": "nnunet-v2", "config": {"modality": "CT"}},
        {"recipe": "monai-unet", "config": {"checkpoint": "/tmp/custom"}},
        {"recipe": "monai-unet", "initial_model_id": "other-project-model"},
    ):
        response = http.post(
            prefix + "/learners", json={"name": "forbidden", "label_ids": [0, 1], **body}
        )
        assert response.status_code == 403, response.text
    assert not service.store.list(Learner)
    assert (
        http.post(
            "/api/projects/not-owned/learners",
            json={"name": "forbidden", "recipe": "monai-unet", "label_ids": [0, 1]},
        ).status_code
        == 403
    )
    with service.store.transaction() as session:
        lease = session.get(DemoAccess, uid)
        session.update(
            lease.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        )
    assert (
        http.post(
            prefix + "/learners",
            json={"name": "expired", "recipe": "monai-unet", "label_ids": [0, 1]},
        ).status_code
        == 401
    )


def test_regular_annotator_does_not_gain_setup_permission(http):
    service = http.app.state.services
    project = http.post("/api/projects", json={"name": "Ordinary project"}).json()
    user = service.auth.create_user("ordinary-annotator", "test-password-1234")
    service.auth.add_member(project["id"], user.id, [Role.ANNOTATOR, Role.REVIEWER])
    http.cookies.clear()
    http.post(
        "/api/auth/login",
        json={"username": user.username, "password": "test-password-1234"},
    ).raise_for_status()
    response = http.post(
        f"/api/projects/{project['id']}/learners",
        json={"name": "Not permitted", "recipe": "monai-unet", "label_ids": [0, 1]},
    )
    assert response.status_code == 403
    assert not service.store.list(Learner)


def test_chat_invalid_base_does_not_add_labels(http, setup_visitor):
    service, pid, _ = setup_visitor
    project = service.store.get(Project, pid)
    wrong = next(m for m in service.store.list(ModelRecord, pid) if m.preset == "vista3d")
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="wrong-base",
                    name="create_learner",
                    arguments={
                        "recipe": "totalsegmentator-ct",
                        "targets": ["Pancreas"],
                        "parent_model_id": wrong.id,
                    },
                )
            ],
        )
    ]
    http.post(f"/api/projects/{pid}/assistant", json={"message": "Create a pancreas setup"})
    assert service.store.get(Project, pid).labels == project.labels
    assert not service.store.list(Learner)
