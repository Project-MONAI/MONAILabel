"""Viewer-independent invalidation follows committed project mutations."""

import pytest

from monailabel.core.models import Asset


def test_project_change_counter_covers_submission_review_and_deletion(client, http, seeded):
    setup, assets = seeded
    project_id = setup["project_id"]
    path = f"/api/projects/{project_id}"

    def version():
        return client.get(path + "/changes")["version"]

    initial = version()
    assert initial > 0
    client.get(path + "/assets")
    client.get(path + "/models")
    client.post("/api/projects", {"name": "Unrelated project"})
    assert version() == initial
    asset = assets[0]
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": 0,
            "mask": client.get(f"/api/assets/{asset['id']}/fixture")["mask"],
            "covered_labels": [0, 1, 2],
        },
    )
    submitted = version()
    assert submitted > initial
    # Acceptance can change status without changing the asset's annotation revision.
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    accepted = version()
    assert accepted > submitted
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 1
    # Failed/rolled-back writes must not invalidate anyone's workspace.
    store = http.app.state.services.store
    with pytest.raises(RuntimeError, match="rollback"), store.transaction() as session:
        record = session.get(Asset, asset["id"])
        session.update(record.model_copy(update={"name": "Not committed"}))
        raise RuntimeError("rollback")
    assert version() == accepted
    deleted = http.request("DELETE", path + "/assets", json={"asset_ids": [asset["id"]]})
    assert deleted.status_code == 200
    assert version() > accepted


def test_project_changes_requires_membership(client, http, seeded):
    setup, _ = seeded
    path = f"/api/projects/{setup['project_id']}/changes"
    client.post("/api/auth/users", {"username": "outsider", "password": "outsider-password"})
    client.post("/api/auth/login", {"username": "outsider", "password": "outsider-password"})
    assert http.get(path).status_code == 403
    client.post("/api/auth/logout")
    assert http.get(path).status_code == 401


def test_review_launch_requires_a_submitted_case_and_reviewer_role(client, http, seeded):
    setup, assets = seeded
    asset = assets[0]
    path = f"/api/assets/{asset['id']}/viewer?name=ohif&mode=review"
    assert http.post(path).status_code == 422
    client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": 0,
            "mask": client.get(f"/api/assets/{asset['id']}/fixture")["mask"],
            "covered_labels": [0, 1, 2],
        },
    )
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "annotator-password"}
    )
    client.request(
        "PUT",
        f"/api/projects/{setup['project_id']}/members",
        {"user_id": user["id"], "roles": ["annotator"]},
    )
    client.post("/api/auth/login", {"username": "annotator", "password": "annotator-password"})
    assert http.post(path).status_code == 403
