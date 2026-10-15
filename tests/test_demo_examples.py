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

from monailabel.core.models import Asset, Job, JobStatus, ModelRecord, Project
from monailabel.core.video import VideoAsset
from monailabel.server.demo_access import DemoAccess
from monailabel.server.demo_context import actor
from monailabel.server.demo_onboarding import TEMPLATES


@pytest.fixture
def example_visitor(http, monkeypatch):
    for name in (
        "SELF_REGISTER",
        "PUBLIC_REGISTER",
        "CACHE_ONLY",
        "TRAINING_PREVIEW",
        "HTTP_LIMITS",
        "LIFECYCLE",
    ):
        monkeypatch.setenv("MONAILABEL_DEMO_" + name, "1")
    service = http.app.state.services
    calls = []

    def start(project_id, request, user_id):
        assert actor.get() == user_id
        assert request.limit == 1 and not request.include_masks
        assert request.template_id in TEMPLATES.values()
        calls.append(request.template_id)
        job = Job(
            project_id=project_id,
            user_id=user_id,
            kind="dataset_import",
            request=request.model_dump(mode="json"),
            status=JobStatus.SUCCEEDED,
        )
        with service.store.transaction() as session:
            session.insert(job)
        return job

    monkeypatch.setattr(service.dataset_templates, "start", start)
    http.cookies.clear()
    credentials = {"username": "example-visitor", "password": "test-password-1234"}
    user = http.post("/api/auth/signup", json=credentials).json()
    http.post("/api/auth/login", json=credentials).raise_for_status()
    project = http.get("/api/projects").json()[0]
    return http, service, user, project, calls


def sample(project_id, example):
    if example == "endoscopy":
        return VideoAsset(
            project_id=project_id,
            name="clip",
            split="pool",
            group_id="hyperkvasir:99b387e7-d07b-4268-9226-4df450c2a198",
            source_key="fixture",
            metadata_key="fixture",
            width=32,
            height=32,
            frames=2,
            duration=1,
        )
    return Asset(
        project_id=project_id,
        name="sample",
        split="pool",
        image_key="fixture",
        group_id=(
            "openslide:CMU-1"
            if example == "pathology"
            else "msd:Task05_Prostate:prostate_00.nii.gz"
            if example == "mri"
            else "msd:Task09_Spleen:spleen_1.nii.gz"
        ),
        spatial_shape=[32, 32] if example == "pathology" else [32, 32, 8],
        kind="image2d" if example == "pathology" else "volume3d",
    )


@pytest.mark.parametrize("example", list(TEMPLATES))
def test_examples_are_fixed_and_idempotent_with_other_samples_present(example_visitor, example):
    http, service, user, project, calls = example_visitor
    with service.store.transaction() as session:
        for other in TEMPLATES:
            if other != example:
                session.insert(sample(project["id"], other))
    response = http.post("/api/demo-access/starter", json={"example": example})
    assert response.status_code == 200, response.text
    assert response.json()["project_id"] == project["id"]
    assert calls[-1] == TEMPLATES[example]
    with service.store.transaction() as session:
        session.insert(sample(project["id"], example))
    before = len(calls)
    assert http.post("/api/demo-access/starter", json={"example": example}).json() is None
    assert len(calls) == before
    roles = http.get(f"/api/projects/{project['id']}/permissions").json()["roles"]
    assert set(roles) == {"annotator", "reviewer"}


def test_examples_reject_arbitrary_project_imports_and_expired_visitors(example_visitor):
    http, service, user, project, calls = example_visitor
    for body in (
        {"example": "external"},
        {"example": "pathology", "project_id": "other"},
        {"example": "endoscopy", "url": "https://example.com"},
        {"example": "radiology", "limit": 100},
    ):
        assert http.post("/api/demo-access/starter", json=body).status_code == 422
    with service.store.transaction() as session:
        lease = session.get(DemoAccess, user["id"])
        session.update(
            lease.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        )
    assert http.post("/api/demo-access/starter", json={"example": "pathology"}).status_code == 401


@pytest.mark.parametrize("example,target", [("pathology", "Nucleus"), ("mri", "Prostate")])
def test_example_keeps_labels_and_adds_target_once(example_visitor, example, target):
    http, service, user, project, calls = example_visitor
    old = service.store.get(Project, project["id"])
    for _ in range(2):
        http.post("/api/demo-access/starter", json={"example": example}).raise_for_status()
    updated = service.store.get(Project, project["id"])
    assert updated.labels[: len(old.labels)] == old.labels
    assert [label.name for label in updated.labels].count(target) == 1
    assert updated.version == old.version + 1


def test_active_import_is_reused_and_other_choices_wait(example_visitor):
    http, service, user, project, calls = example_visitor
    with service.store.transaction() as session:
        job = session.list(Job, project["id"])[0]
        session.update(job.model_copy(update={"status": JobStatus.RUNNING}))
    before = len(calls)
    assert http.post("/api/demo-access/starter").json()["id"] == job.id
    assert http.post("/api/demo-access/starter", json={"example": "pathology"}).status_code == 409
    assert len(calls) == before


def test_retry_budget_is_per_example_and_interactive_models_are_registered(example_visitor):
    http, service, user, project, calls = example_visitor
    for _ in range(3):
        http.post("/api/demo-access/starter", json={"example": "endoscopy"}).raise_for_status()
    assert http.post("/api/demo-access/starter", json={"example": "endoscopy"}).status_code == 409
    http.post("/api/demo-access/starter", json={"example": "pathology"}).raise_for_status()
    models = service.store.list(ModelRecord, project["id"])
    assert {m.provider for m in models} >= {"sam2", "nninteractive"}


def test_starter_requires_authentication(http):
    http.cookies.clear()
    assert http.post("/api/demo-access/starter", json={"example": "pathology"}).status_code == 401
