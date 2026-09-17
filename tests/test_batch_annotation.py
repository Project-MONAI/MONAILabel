"""Batch inference preserves held-out references and requires real human decisions."""

from threading import Event

import numpy as np
import pytest
from test_coordinator import tool

from monailabel.core.errors import DomainError
from monailabel.core.evaluation import EvaluationReservation
from monailabel.core.models import (
    Annotation,
    Asset,
    BatchAnnotateRequest,
    Job,
    JobStatus,
    Proposal,
    ReviewDecision,
    ReviewRequest,
)


def test_chat_batch_runs_first_five_and_submits_only_requested_label(http, client, seeded):
    setup, assets = seeded
    service = http.app.state.services
    service.assistants.provider.queue = [
        tool(
            "annotate_batch",
            targets=["Structure A"],
            model_name="Demo intensity thresholds",
            limit=5,
            submit_for_review=True,
        )
    ]
    body = {
        "message": "Segment Structure A in the first 5 images and submit them for review",
        "request_id": "a" * 32,
    }
    path = f"/api/projects/{setup['project_id']}/assistant"
    reply = client.post(path, body)
    result = client.wait(reply["job_id"])
    assert result["asset_ids"] == [a["id"] for a in assets[:5]]
    assert len(result["annotation_ids"]) == 5
    assert result["failed"] == []
    logs = client.get(f"/api/jobs/{reply['job_id']}/logs")
    assert any("Image 1 of 5" in e["message"] for e in logs["entries"])
    assert sum("submitted for pending review" in e["message"] for e in logs["entries"]) == 5
    download = http.get(f"/api/jobs/{reply['job_id']}/logs/download")
    assert "batch-segmentation-" in download.headers["content-disposition"]
    assert "Batch segmentation completed." in download.text
    for identifier in result["annotation_ids"]:
        annotation = service.store.get(Annotation, identifier)
        proposal = service.store.get(Proposal, annotation.proposal_id)
        assert annotation.covered_labels == [0, 1]
        assert proposal.label_ids == [1]
        assert proposal.model_ids == [setup["baseline_id"]]
        assert set(np.unique(service.artifacts.array(annotation.mask_key))) == {0, 1}
        assert annotation.reviewer == client.get("/api/auth/me")["id"]
    assert service.store.list(ReviewDecision, setup["project_id"]) == []
    assert http.post(f"/api/projects/{setup['project_id']}/snapshots").status_code == 422
    assert client.post(path, body) == reply
    assert len(service.store.list(Annotation, setup["project_id"])) == 5


def test_batch_skips_evaluation_aliases_saved_annotations_and_pending_proposals(
    http, client, seeded
):
    setup, assets = seeded
    service = http.app.state.services
    store = service.store
    with store.transaction() as session:
        session.insert(
            EvaluationReservation(
                project_id=setup["project_id"],
                group_id=assets[0]["group_id"],
                image_keys=[assets[1]["image_key"]],
            )
        )
    proposal = client.wait(client.post(f"/api/assets/{assets[2]['id']}/annotate", {})["id"])
    client.post(
        f"/api/assets/{assets[2]['id']}/review",
        {
            "base_revision": 0,
            "proposal_id": proposal["proposal_id"],
            "covered_labels": [0, 1, 2],
        },
    )
    client.wait(client.post(f"/api/assets/{assets[3]['id']}/annotate", {})["id"])
    job = service.batch_annotations.start(
        setup["project_id"], BatchAnnotateRequest(limit=100), "test"
    )
    result = client.wait(job.id)
    assert result["asset_ids"] == [a["id"] for a in assets[4:6] + assets[9:]]
    assert result["annotation_ids"] == []
    assert len(store.list(Annotation, setup["project_id"])) == 1


def test_batch_preserves_concurrent_edit_and_reports_partial_failure(
    http, client, seeded, monkeypatch
):
    setup, assets = seeded
    service = http.app.state.services
    original = service.models.predict
    calls = 0

    def predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            service.annotations.review(
                assets[0]["id"],
                ReviewRequest(
                    base_revision=0,
                    covered_labels=[0, 1, 2],
                    reviewer="human",
                ),
                imported_mask=np.zeros(assets[0]["spatial_shape"], dtype=np.uint8),
            )
        if calls == 2:
            raise DomainError("Provider unavailable for this case")
        return original(*args, **kwargs)

    monkeypatch.setattr(service.models, "predict", predict)
    job = service.batch_annotations.start(
        setup["project_id"],
        BatchAnnotateRequest(
            limit=3,
            submit_for_review=True,
        ),
        "test",
    )
    result = client.wait(job.id)
    assert result["asset_ids"] == [assets[2]["id"]]
    assert len(result["failed"]) == 2
    first = service.store.get(Asset, assets[0]["id"])
    assert first.revision == 1
    assert service.store.get(Annotation, first.annotation_id).reviewer == "human"
    assert service.store.get(Asset, assets[1]["id"]).annotation_id is None


def test_cancel_batch_retains_completed_cases_without_publishing_inflight_case(
    http, client, seeded, monkeypatch
):
    setup, assets = seeded
    service = http.app.state.services
    original = service.models.predict
    entered, release = Event(), Event()
    calls = 0

    def predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.models, "predict", predict)
    job = service.batch_annotations.start(
        setup["project_id"],
        BatchAnnotateRequest(
            limit=3,
            submit_for_review=True,
        ),
        "test",
    )
    try:
        assert entered.wait(10)
        running = service.store.get(Job, job.id)
        assert 0 < running.progress < 1
        assert "Image 2 of 3" in running.progress_message
        assert any("segmentation started" in e.message for e in service.jobs.logs(job.id).entries)
        # Concurrent batches cannot select cases already claimed by this job.
        other = service.batch_annotations.start(
            setup["project_id"], BatchAnnotateRequest(limit=1), "test"
        )
        assert other.request["asset_ids"] == [assets[3]["id"]]
        service.jobs.cancel(other.id)
        service.jobs.cancel(job.id)
    finally:
        release.set()
    service.jobs.executor.shutdown(wait=True)
    saved = service.store.get(Job, job.id)
    assert saved.status == JobStatus.CANCELLED
    assert saved.result["asset_ids"] == [assets[0]["id"]]
    assert len(service.store.list(Annotation, setup["project_id"])) == 1
    assert not any(
        p.asset_id in {assets[1]["id"], assets[2]["id"]}
        for p in service.store.list(Proposal, setup["project_id"])
    )


def test_batch_validates_before_starting_and_requires_annotation_permission(http, client, seeded):
    setup, _ = seeded
    service = http.app.state.services
    with pytest.raises(DomainError):
        service.batch_annotations.start(
            setup["project_id"],
            BatchAnnotateRequest(
                model_id=setup["baseline_id"],
                label_ids=[99],
            ),
            "test",
        )
    assert service.store.list(Job, setup["project_id"]) == []
    user = client.post(
        "/api/auth/users", {"username": "reviewer", "password": "test-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{setup['project_id']}/members",
        {
            "user_id": user["id"],
            "roles": ["reviewer"],
        },
    )
    client.post("/api/auth/login", {"username": "reviewer", "password": "test-password-1234"})
    service.assistants.provider.queue = [tool("annotate_batch", limit=5, submit_for_review=True)]
    response = http.post(
        f"/api/projects/{setup['project_id']}/assistant", json={"message": "Segment five"}
    )
    assert response.status_code == 403
    assert service.store.list(Job, setup["project_id"]) == []


def test_batch_all_failures_are_visible_in_result_and_logs(http, client, seeded, monkeypatch):
    from test_jobs import wait

    setup, _ = seeded
    service = http.app.state.services

    def fail(*args, **kwargs):
        raise DomainError("Model unavailable")

    monkeypatch.setattr(service.models, "predict", fail)
    job = service.batch_annotations.start(
        setup["project_id"],
        BatchAnnotateRequest(
            limit=2,
            submit_for_review=True,
        ),
        "test",
    )
    saved = wait(service.store, job.id)
    assert saved.status == JobStatus.FAILED
    assert saved.result["annotation_ids"] == []
    assert len(saved.result["failed"]) == 2
    assert sum(e.level == "error" for e in service.jobs.logs(job.id).entries) == 3
    assert service.store.list(Annotation, setup["project_id"]) == []


def test_explicit_vista_spleen_batch_overrides_selected_model(http, client, seeded, monkeypatch):
    from monailabel.core.models import Label, ModelRecord, Project

    setup, _ = seeded
    service = http.app.state.services
    project = service.store.get(Project, setup["project_id"])
    vista = ModelRecord(
        project_id=project.id,
        name="VISTA3D",
        provider="vista3d",
        read_only=True,
        label_ids=[0],
    )
    with service.store.transaction() as session:
        session.insert(vista)
        session.update(
            project.model_copy(
                update={
                    "labels": [
                        Label(id=0, name="Background"),
                        Label(id=1, name="Spleen"),
                        Label(id=2, name="Liver"),
                    ]
                }
            )
        )
    calls = []

    def predict(project, model, image, prompt, affine):
        assert model.id == vista.id
        assert model.label_ids == [0, 1]
        calls.append(model.id)
        mask = np.zeros(image.shape[:3], dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        return mask

    monkeypatch.setattr(service.models, "predict", predict)
    service.assistants.provider.queue = [
        tool(
            "annotate_batch",
            targets=["spleen"],
            model_name="VISTA3D",
            limit=5,
            submit_for_review=True,
        )
    ]
    reply = client.post(
        f"/api/projects/{project.id}/assistant",
        {
            "message": (
                "Run VISTA3D segmentation to annotate spleen for first 5 images "
                "and submit them for review."
            ),
            "context": {"model_id": setup["baseline_id"]},
        },
    )
    result = client.wait(reply["job_id"])
    assert len(calls) == len(result["annotation_ids"]) == 5
    assert not result["failed"]
    assert all(p.label_ids == [1] for p in service.store.list(Proposal, project.id))
    assert service.store.list(ReviewDecision, project.id) == []
