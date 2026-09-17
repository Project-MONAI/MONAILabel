"""A manager's explicit choice preserves teacher provenance and held-out review."""

import pytest

from monailabel.core.models import ReviewDecision, Snapshot


def submit_prediction(client, asset, scope=None):
    request = {"slice": scope} if scope else {}
    result = client.wait(client.post(f"/api/assets/{asset['id']}/annotate", request)["id"])
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "proposal_id": result["proposal_id"], "covered_labels": [0, 1, 2]},
    )
    return annotation


@pytest.mark.parametrize("note", [None, "Owner-authorized teacher experiment"])
def test_explicit_prediction_snapshot_records_origin_without_review_and_excludes_validation(
    client, http, seeded, note
):
    setup, assets = seeded
    prefix = f"/api/projects/{setup['project_id']}"
    train = next(a for a in assets if a["split"] == "train")
    validation = next(a for a in assets if a["split"] == "validation")
    original = submit_prediction(client, train)
    submit_prediction(client, validation)
    body = {"allow_unreviewed_training": True}
    if note is not None:
        body["note"] = note
    assert http.post(prefix + "/snapshots").status_code == 422
    snapshot = client.post(prefix + "/snapshots", body)
    assert snapshot["note"] == (note or "")
    assert len(snapshot["samples"]) == 1
    sample = snapshot["samples"][0]
    assert sample["asset_id"] == train["id"]
    assert sample["decision_id"] is None
    assert sample["label_source"] == "model_prediction"
    assert sample["proposal_id"] == original["proposal_id"]
    assert sample["model_ids"]
    assert snapshot["authorized_by"] == client.get("/api/auth/me")["id"]
    assert http.app.state.services.store.list(ReviewDecision, setup["project_id"]) == []
    # Explicitly rejected predictions cannot enter a subsequent snapshot.
    client.post(f"/api/annotations/{original['id']}/decision", {"verdict": "changes_requested"})
    assert http.post(prefix + "/snapshots", json=body).status_code == 422
    saved = http.app.state.services.store.get(Snapshot, snapshot["id"])
    assert saved.samples[0].decision_id is None


def test_partial_or_manual_drafts_cannot_bypass_review(client, http, seeded):
    setup, assets = seeded
    trains = [a for a in assets if a["split"] == "train"]
    submit_prediction(client, trains[0], {"axis": 2, "index": 2})
    mask = client.get(f"/api/assets/{trains[1]['id']}/fixture")["mask"]
    client.post(
        f"/api/assets/{trains[1]['id']}/review",
        {"base_revision": 0, "covered_labels": [0, 1, 2], "mask": mask},
    )
    assert (
        http.post(
            f"/api/projects/{setup['project_id']}/snapshots",
            json={"allow_unreviewed_training": True, "note": "Experiment"},
        ).status_code
        == 422
    )


def test_prediction_training_keeps_validation_separate_and_marks_model(client, http, seeded):
    setup, assets = seeded
    prefix = f"/api/projects/{setup['project_id']}"
    train = next(a for a in assets if a["split"] == "train")
    validation = next(a for a in assets if a["split"] == "validation")
    submit_prediction(client, train)
    reference = client.get(f"/api/assets/{validation['id']}/fixture")["mask"]
    annotation = client.post(
        f"/api/assets/{validation['id']}/review",
        {"base_revision": 0, "mask": reference, "covered_labels": [0, 1, 2]},
    )
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    learner = client.post(
        prefix + "/learners",
        {"name": "Teacher student", "recipe": "pixel-gaussian", "label_ids": [0, 1, 2]},
    )
    job = client.post(
        prefix + f"/learners/{learner['id']}/train",
        {"allow_unreviewed_training": True},
    )
    result = client.wait(job["id"])
    model = next(m for m in client.get(prefix + "/models") if m["id"] == result["model_id"])
    assert model["unreviewed_training"] is True
    assert model["training_assets"] == [train["id"]]
    snapshot = http.app.state.services.store.get(Snapshot, result["snapshot_id"])
    assert snapshot.note == ""
    assert snapshot.authorized_by == client.get("/api/auth/me")["id"]
    assert {s.label_source for s in snapshot.samples} == {"model_prediction", "reviewed"}
    result = client.wait(
        client.post(
            prefix + "/evaluate",
            {
                "snapshot_id": snapshot.id,
                "candidate_id": model["id"],
                "baseline_id": setup["baseline_id"],
            },
        )["id"]
    )
    assert result["evaluation_id"]


def test_prediction_exception_requires_manager(client, http, seeded):
    setup, _ = seeded
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "annotator-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{setup['project_id']}/members",
        {"user_id": user["id"], "roles": ["annotator"]},
    )
    client.post("/api/auth/login", {"username": "annotator", "password": "annotator-password-1234"})
    assert (
        http.post(
            f"/api/projects/{setup['project_id']}/snapshots",
            json={"allow_unreviewed_training": True},
        ).status_code
        == 403
    )
