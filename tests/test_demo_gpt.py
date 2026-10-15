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
from unittest.mock import Mock

import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, ModelRecord, Project
from monailabel.core.ports import Prediction
from monailabel.server.demo_access import DemoAccess
from monailabel.server.demo_compute import execute
from monailabel.server.demo_context import acting_as
from monailabel.server.demo_gpt import MODELS, approved, ensure


@pytest.fixture
def gpt_visitor(http, monkeypatch):
    for suffix in (
        "SELF_REGISTER",
        "PUBLIC_REGISTER",
        "HTTP_LIMITS",
        "LIFECYCLE",
        "CACHE_ONLY",
        "TRAINING_PREVIEW",
        "GPT",
    ):
        monkeypatch.setenv("MONAILABEL_DEMO_" + suffix, "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fixture-not-a-real-key")
    service = http.app.state.services
    monkeypatch.setattr(service.demo_onboarding, "start", lambda *args: None)
    http.cookies.clear()
    credentials = {"username": "gpt-visitor", "password": "test-password-1234"}
    registered = http.post("/api/auth/signup", json=credentials)
    assert registered.status_code == 201, registered.text
    user = registered.json()
    http.post("/api/auth/login", json=credentials).raise_for_status()
    project = service.store.list(Project)[0]
    models = [m for m in service.store.list(ModelRecord, project.id) if m.preset in MODELS]
    return service, user["id"], project, models


def test_new_and_existing_projects_get_exact_gpt_choices_without_default_changes(http, gpt_visitor):
    service, user, project, models = gpt_visitor
    assert len(models) == 2 and all(approved(m) for m in models)
    before = service.store.list(ModelRecord, project.id)
    service.presets.ensure(project.id)
    assert service.store.list(ModelRecord, project.id) == before
    assert service.store.get(Project, project.id) == project
    visible = http.get(f"/api/projects/{project.id}/models")
    assert visible.status_code == 200
    assert {m.id for m in models} <= {m["id"] for m in visible.json()}
    assert "sk-fixture" not in visible.text
    assert "OpenAI" in http.get("/api/auth/status").json()["activity_notice"]
    with service.store.transaction() as session:
        existing = Project(name="Existing operator project")
        session.insert(existing)
    service.presets.ensure(existing.id)
    assert service.store.get(Project, existing.id) == existing
    assert len(service.store.list(ModelRecord, existing.id)) == 2
    with acting_as(user), pytest.raises(DomainError):
        service.models.get(project.id, service.store.list(ModelRecord, existing.id)[0].id)


@pytest.mark.parametrize("missing", ["MONAILABEL_DEMO_GPT", "OPENAI_API_KEY"])
def test_opt_in_and_key_are_both_required_at_each_inference(gpt_visitor, monkeypatch, missing):
    service, user, project, models = gpt_visitor
    monkeypatch.delenv(missing)
    with acting_as(user):
        assert not any(approved(m) for m in models)
        assert not {m.id for m in models} & {m.id for m in service.models.available(project.id)}
        with pytest.raises(DomainError, match="managed by the demo host"):
            service.models.get(project.id, models[0].id)


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://other.invalid/v1/responses"},
        {"model": "not-approved"},
        {"token_env": "ANOTHER_KEY"},
        {"credential_id": "saved-key"},
        {"max_output_tokens": 16384},
        {"reasoning_effort": "max"},
    ],
)
def test_visitor_cannot_reuse_approval_for_a_different_connection(gpt_visitor, change):
    service, user, project, models = gpt_visitor
    altered = models[0].model_copy(update={"config": models[0].config | change})
    with acting_as(user), pytest.raises(DomainError):
        service.models.require_visitor_model(altered)
    assert not approved(altered)


def test_permissions_expiry_and_provider_configuration_remain_enforced(http, gpt_visitor):
    service, user, project, models = gpt_visitor
    assert set(http.get(f"/api/projects/{project.id}/permissions").json()["roles"]) == {
        "annotator",
        "reviewer",
    }
    for route, body in (
        ("models", {"name": "Other", "provider": "http-mask"}),
        ("credentials", {"name": "Key", "api_key": "not-a-secret"}),
        ("model-catalog", {"service": "openai"}),
    ):
        assert http.post(f"/api/projects/{project.id}/{route}", json=body).status_code == 403
    with service.store.transaction() as session:
        lease = session.get(DemoAccess, user)
        session.update(
            lease.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        )
    assert http.get(f"/api/projects/{project.id}/models").status_code == 401
    http.cookies.clear()
    assert http.get(f"/api/projects/{project.id}/models").status_code == 401


def test_timed_job_and_worker_allow_only_canonical_gpt(gpt_visitor, monkeypatch):
    service, user, project, models = gpt_visitor
    model = service.models.for_labels(models[0], [1])
    image = np.zeros((4, 5, 3), dtype=np.float32)
    mask = np.ones((4, 5), dtype=np.uint8)
    isolated = Mock(return_value=mask)
    monkeypatch.setattr("monailabel.server.models.service.timed_compute", lambda: True)
    monkeypatch.setattr("monailabel.server.models.service.isolated", isolated)
    with acting_as(user):
        np.testing.assert_array_equal(service.models.predict(project, model, image), mask)
    assert isolated.call_args.args[0] == "predict"
    monkeypatch.setattr("monailabel.server.models.service.timed_compute", lambda: False)
    monkeypatch.setenv("MONAILABEL_PREPARED_MODELS_ONLY", "1")
    predict = Mock(return_value=Prediction(mask))
    monkeypatch.setattr("monailabel.providers.remote.RemoteSegmenter.predict", predict)
    # Fresh worker has no request identity, but retains the same exact connection gate.
    np.testing.assert_array_equal(
        execute(
            "predict", (service.store.path, service.artifacts.root, project, model, image, "", None)
        ),
        mask,
    )
    assert predict.call_count == 1
    monkeypatch.delenv("MONAILABEL_DEMO_GPT")
    with pytest.raises(DomainError):
        execute(
            "predict", (service.store.path, service.artifacts.root, project, model, image, "", None)
        )
    assert predict.call_count == 1


def test_visitor_http_annotation_publishes_draft_without_changing_source(
    http, client, gpt_visitor, monkeypatch
):
    service, user, project, models = gpt_visitor
    source = np.zeros((4, 5, 3), dtype=np.float32)
    asset = Asset(
        project_id=project.id,
        name="Synthetic GPT fixture",
        kind="image2d",
        split="pool",
        spatial_shape=[4, 5],
        image_key=service.artifacts.put_array(source),
        group_id="synthetic",
    )
    with service.store.transaction() as session:
        session.insert(asset)
    calls = []

    def worker(operation, database, root, requested_project, model, image, prompt, affine):
        assert operation == "predict" and requested_project.id == project.id
        assert approved(model) and model.label_ids == [0, 1]
        calls.append(model.config["model"])
        return np.ones((4, 5), dtype=np.uint8)

    monkeypatch.setattr("monailabel.server.models.service.isolated", worker)
    job = http.post(
        f"/api/assets/{asset.id}/annotate", json={"model_id": models[0].id, "label_ids": [1]}
    )
    assert job.status_code == 202, job.text
    result = client.wait(job.json()["id"])
    proposal = http.get(f"/api/proposals/{result['proposal_id']}").json()
    assert proposal["base_revision"] == 0
    assert service.store.get(Asset, asset.id).revision == 0
    np.testing.assert_array_equal(service.artifacts.array(asset.image_key), source)
    assert calls == [models[0].config["model"]]


def test_approval_does_not_depend_on_a_username(http, gpt_visitor):
    service, _, _, _ = gpt_visitor
    other = service.auth.create_user("any-registered-user", "test-password-1234", demo_minutes=60)
    with service.store.transaction() as session:
        project = Project(name="Another registered user")
        session.insert(project)
        ensure(session, project.id)
    with acting_as(other.id):
        assert len([m for m in service.models.available(project.id) if approved(m)]) == 2
