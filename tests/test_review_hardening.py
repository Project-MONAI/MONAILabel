"""Regression coverage for model lineage, atomic job guards and scoped chat reviews."""

import runpy
from pathlib import Path

import pytest
from test_model_splits import add_cases, setup, train

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import Job, ModelRecord, Project, Snapshot, Split
from monailabel.server.assistant_tools.base import ToolContext


def test_exact_spleen_golden_workflow():
    script = Path(__file__).parents[1] / "examples/verify_spleen_workflow.py"
    report = runpy.run_path(str(script))["run_workflow"]()
    assert len(report["steps"]) == 8
    assert report["counts"] == {
        "annotation_images": 32,
        "evaluation_images": 9,
        "trained_images": 5,
    }
    assert report["real_vista3d"] is False and report["real_coordinator"] is False


def test_evaluation_review_scope_does_not_accept_annotation_samples(client, http):
    prefix = setup(client)
    assets = add_cases(client, http, prefix, 0, 4, accepted=False)
    client.post(
        prefix + "/evaluation-sets",
        {
            "name": "Fixed",
            "percentage": 100,
            "auto_update": False,
            "asset_ids": [a["id"] for a in assets[:2]],
        },
    )
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="review",
                    name="review_saved_annotations",
                    arguments={
                        "scope": "pending",
                        "verdict": "accepted",
                        "dataset_use": "evaluation",
                    },
                )
            ],
        )
    ]
    client.post(prefix + "/assistant", {"message": "Mark evaluation reviews as good"})
    assert {d["asset_id"] for d in client.get(prefix + "/decisions")} == {
        a["id"] for a in assets[:2]
    }


def test_training_job_guard_rechecks_active_learner_atomically(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 5)
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Student",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    model, _ = train(client, prefix, learner, 20)
    with http.app.state.services.store.transaction() as session:
        session.insert(
            Job(
                kind="train",
                project_id=prefix.split("/")[-1],
                request={"learner_id": learner["id"]},
                status="running",
            )
        )
    response = http.post(
        prefix + "/train",
        json={
            "learner_id": learner["id"],
            "snapshot_id": model["snapshot_id"],
            "label_ids": [0, 1],
        },
    )
    assert response.status_code == 409 and "already training" in response.text
    assert len(client.get(prefix + "/jobs")) == 2


def test_ancestor_image_alias_cannot_enter_evaluation(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 6)
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Student",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    first, _ = train(client, prefix, learner, 20)
    services = http.app.state.services
    store = services.store
    initial = store.get(Snapshot, first["snapshot_id"])
    old_sample = next(s for s in initial.samples if s.split == Split.TRAIN)
    # A descendant's current snapshot need not contain every image seen by its parents.
    current = initial.model_copy(
        update={
            "id": "new-snapshot",
            "samples": [s for s in initial.samples if s.asset_id != old_sample.asset_id],
        }
    )
    parent = store.get(ModelRecord, first["id"])
    child = parent.model_copy(
        update={"id": "child", "snapshot_id": current.id, "parent_id": parent.id}
    )
    with store.transaction() as session:
        session.insert(current)
        session.insert(child)
    alias = old_sample.model_copy(
        update={"asset_id": "alias", "group_id": "different-patient", "split": Split.VALIDATION}
    )
    references = initial.model_copy(update={"samples": [alias]})
    with pytest.raises(DomainError, match="overlap"):
        services.scorer.samples(store.get(Project, prefix.split("/")[-1]), child, references, [1])


def test_model_name_selects_latest_family_version_but_explicit_id_selects_exact(client, http):
    from monailabel.core.models import AssistantContext, User

    prefix = setup(client)
    pid = prefix.split("/")[-1]
    services = http.app.state.services
    user = services.store.list(User)[0]
    context = ToolContext(services, pid, user, AssistantContext(), "Compare Student")
    records = [
        ModelRecord(
            project_id=pid,
            name="Student",
            provider="pixel-gaussian",
            label_ids=[0, 1],
            learner_id="one-family",
        )
        for _ in range(2)
    ]
    with services.store.transaction() as session:
        for record in records:
            session.insert(record)
    assert context.model_id(None, "Student") == records[-1].id
    assert context.model_id(records[0].id, "Student") == records[0].id
    with services.store.transaction() as session:
        session.insert(
            records[0].model_copy(update={"id": "different", "learner_id": "other-family"})
        )
    with pytest.raises(DomainError, match="exact model name"):
        context.model_id(None, "Student")


@pytest.mark.parametrize(
    "fields",
    [
        {"evaluation_set_id": "set", "snapshot_id": "snapshot"},
        {"evaluation_version_id": "version", "snapshot_id": "snapshot"},
        {"evaluation_set_id": "set", "evaluation_version_id": "version"},
    ],
)
def test_comparison_rejects_ambiguous_references(fields):
    from pydantic import ValidationError

    from monailabel.core.models import EvaluateRequest

    with pytest.raises(ValidationError, match="Choose one"):
        EvaluateRequest(candidate_id="new", baseline_id="base", **fields)


def test_failed_startup_closes_workers_and_releases_workspace(tmp_path, monkeypatch):
    from filelock import FileLock

    from monailabel.server import service

    closed = []
    original = service.Jobs.close

    def close(jobs):
        closed.append(jobs)
        original(jobs)

    def fail(runtime):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(service.Jobs, "close", close)
    monkeypatch.setattr(service.CoordinatorRuntime, "start", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        service.Services(tmp_path)
    assert len(closed) == 1
    with FileLock(str(tmp_path / "server.lock"), timeout=0):
        pass


@pytest.mark.parametrize("labels", [False, True])
def test_oversized_image_reports_bounded_error(monkeypatch, labels):
    import io

    from PIL import Image

    from monailabel.server.data import decode_image
    from monailabel.server.reference_imports import decode_labels

    content = io.BytesIO()
    Image.new("L", (10, 10)).save(content, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 20)
    with pytest.raises(DomainError) as error:
        (decode_labels if labels else decode_image)("large.png", content.getvalue())
    assert error.value.status == 413


def test_workspace_tools_omit_image_edits_until_a_sample_is_selected(client, http, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    for context in ({}, {"asset_id": assets[0]["id"]}):
        planner.queue = [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="skill",
                        name="load_skill",
                        arguments={"name": "monailabel-review", "purpose": "answer"},
                    )
                ],
            ),
            ChatMessage(role="assistant", content="Ready"),
        ]
        client.post(
            f"/api/projects/{setup['project_id']}/assistant",
            {"message": "What can I do here?", "context": context},
        )
        names = {tool.name for tool in planner.calls[-1][1]}
        assert {"load_skill", "inspect_workspace", "review_saved_annotations"} <= names
        assert not {"annotate_batch", "start_training", "edit_spatial_prompts"} & names
        assert ("review_annotation" in names) == bool(context)


def test_vista_learner_uses_pretrained_recipe_default(client, http):
    services = http.app.state.services
    services.presets.enabled = True
    project = client.post("/api/projects", {"name": "Spleen model"})
    services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="create",
                    name="create_learner",
                    arguments={"recipe": "vista3d", "name": "Spleen", "targets": ["Spleen"]},
                )
            ],
        )
    ]
    reply = client.post(
        f"/api/projects/{project['id']}/assistant", {"message": "Create a VISTA3D spleen model"}
    )
    learner = client.get(f"/api/projects/{project['id']}/learners")[0]
    base = services.store.get(ModelRecord, learner["initial_model_id"])
    assert base.read_only and base.preset == "vista3d"
    assert reply["job_id"] is None
