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

import json
from datetime import UTC, datetime

import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import Job, Snapshot, StartTraining, TrainRequest
from monailabel.server.demo_training import RecordedTraining, recorded_example
from monailabel.server.jobs import Outcome


def example():
    return RecordedTraining(
        recipe="nnunet-v2",
        name="Test fixture, not a real run",
        dataset="Synthetic test fixture",
        completed_at=datetime.now(UTC),
        duration_seconds=2,
        training_cases=2,
        evaluation_cases=1,
        epochs=1,
        steps_per_epoch=2,
        checkpoint_sha256="a" * 64,
        source_revision="b" * 40,
        caveats=["Automated test fixture only."],
        log="fixture log\n",
    )


def test_blocked_training_does_not_create_jobs_or_snapshots(client, http, monkeypatch):
    project = client.post(
        "/api/projects",
        {
            "name": "Private project",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Spleen"},
            ],
        },
    )
    learner = client.post(
        f"/api/projects/{project['id']}/learners",
        {
            "name": "Test",
            "recipe": "monai-unet",
            "label_ids": [0, 1],
        },
    )
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_PREVIEW", "1")
    assert client.get("/api/auth/status")["training_preview"] is True
    service = http.app.state.services
    for route, body in (
        (f"/api/projects/{project['id']}/learners/{learner['id']}/train", {}),
        (f"/api/projects/{project['id']}/train", {"snapshot_id": "missing"}),
    ):
        response = http.post(route, json=body)
        assert response.status_code == 403, response.text
        assert response.json()["code"] == "demo_training_preview"
    with pytest.raises(DomainError, match="no training will run"):
        service.learning.start(project["id"], learner["id"], StartTraining())
    with pytest.raises(DomainError, match="no training will run"):
        service.learning.train(project["id"], TrainRequest(snapshot_id="missing"))
    with pytest.raises(DomainError, match="no training will run"):
        service.jobs.submit("train", project["id"], {}, lambda _: pytest.fail("Must not execute"))
    assert service.store.list(Job) == []
    assert service.store.list(Snapshot) == []
    # Other work is not disabled by the training-only policy.
    job = service.jobs.submit("fixture", project["id"], {}, lambda _: Outcome({"ok": True}))
    assert client.wait(job.id) == {"ok": True}


def test_chat_training_returns_explicit_preview_not_a_job(client, http, monkeypatch):
    project = client.post("/api/projects", {"name": "Private project"})
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_PREVIEW", "1")
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="preview", name="start_training", arguments={})],
        )
    ]
    reply = client.post(f"/api/projects/{project['id']}/assistant", {"message": "Train my model."})
    assert reply["job_id"] is None
    assert reply["data"]["job_started"] is False
    assert "Your current corrections were not used" in reply["message"]
    assert http.app.state.services.store.list(Job) == []
    assert http.app.state.services.store.list(Snapshot) == []


@pytest.mark.parametrize("mode", ["scratch", "fine_tune", "continue"])
def test_every_training_mode_is_blocked(client, http, monkeypatch, mode):
    project = client.post("/api/projects", {"name": "Training disabled"})
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_PREVIEW", "1")
    response = http.post(
        f"/api/projects/{project['id']}/learners/not-created/train",
        json={
            "mode": mode,
            "parent_model_id": None if mode == "scratch" else "parent",
        },
    )
    assert response.status_code == 403
    assert response.json()["code"] == "demo_training_preview"
    assert http.app.state.services.store.list(Job) == []


def test_create_and_train_chat_only_creates_a_setup(client, http, monkeypatch):
    project = client.post(
        "/api/projects",
        {
            "name": "Training disabled",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Spleen"},
            ],
        },
    )
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_PREVIEW", "1")
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="create",
                    name="create_learner",
                    arguments={
                        "recipe": "nnunet-v2",
                        "modality": "CT",
                        "targets": ["spleen"],
                        "name": "Visitor setup",
                        "start_now": True,
                    },
                ),
            ],
        )
    ]
    result = client.post(
        f"/api/projects/{project['id']}/assistant",
        {"message": "Create and train an nnU-Net for spleen."},
    )
    assert result["job_id"] is None
    assert "no training will run" in result["message"]
    assert http.app.state.services.store.list(Job) == []
    assert http.app.state.services.store.list(Snapshot) == []


def test_recorded_catalog_is_authenticated_read_only_and_separate(
    client, http, monkeypatch, tmp_path
):
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_EXAMPLES", str(tmp_path))
    path = tmp_path / "nnunet-v2.json"
    path.write_text(example().model_dump_json())
    assert client.get("/api/demo-access/training")["job_started"] is False
    assert client.get("/api/demo-access/training/nnunet-v2")["presentation"] == "recorded_run"
    assert http.get("/api/demo-access/training/vista3d").status_code == 404
    assert http.post("/api/demo-access/training/nnunet-v2", json={}).status_code == 405
    assert http.app.state.services.store.list(Job) == []
    http.cookies.clear()
    assert http.get("/api/demo-access/training/nnunet-v2").status_code == 401


@pytest.mark.parametrize(
    "invalid", ["wrong_recipe", "invalid_json", "oversized", "symlink", "nan", "directory"]
)
def test_invalid_recorded_examples_fail_closed(tmp_path, monkeypatch, invalid):
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_EXAMPLES", str(tmp_path))
    path = tmp_path / "nnunet-v2.json"
    payload = example().model_dump(mode="json")
    if invalid == "wrong_recipe":
        payload["recipe"] = "vista3d"
    if invalid == "nan":
        payload["final_loss"] = float("nan")
    path.write_text(json.dumps(payload))
    if invalid == "invalid_json":
        path.write_text("not json")
    elif invalid == "oversized":
        path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    elif invalid == "symlink":
        target = tmp_path / "private.json"
        path.rename(target)
        path.symlink_to(target)
    elif invalid == "directory":
        path.unlink()
        path.mkdir()
    with pytest.raises(DomainError) as error:
        recorded_example("nnunet-v2")
    assert error.value.status == 503
    with pytest.raises(DomainError) as error:
        recorded_example("../private")
    assert error.value.status == 404
