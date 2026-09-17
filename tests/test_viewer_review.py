import json

import numpy as np
import pytest


def submitted(client, seeded):
    setup, assets = seeded
    asset = assets[0]
    mask = np.asarray(client.get(f"/api/assets/{asset['id']}/fixture")["mask"], dtype=np.uint8)
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": mask.tolist(), "covered_labels": [0, 1, 2]},
    )
    return setup, asset, mask, annotation


def complete(http, asset, mask, revision=1, labels=None):
    return http.post(
        f"/api/assets/{asset['id']}/review-complete",
        content=mask.tobytes(),
        headers={
            "X-MONAILABEL-REVIEW": json.dumps(
                {
                    "base_revision": revision,
                    "covered_labels": labels or [0, 1, 2],
                    "comment": "Checked in viewer",
                }
            )
        },
    )


def test_good_without_changes_accepts_the_existing_revision(http, client, seeded):
    setup, asset, mask, annotation = submitted(client, seeded)
    result = complete(http, asset, mask)
    assert result.status_code == 201
    assert result.json()["id"] == annotation["id"]
    decisions = client.get(f"/api/projects/{setup['project_id']}/decisions")
    assert (
        decisions[-1]["annotation_id"] == annotation["id"]
        and decisions[-1]["verdict"] == "accepted"
    )
    assert len(client.get(f"/api/assets/{asset['id']}/annotations")) == 1


def test_reviewer_corrections_and_good_are_saved_together(http, client, seeded):
    setup, asset, mask, original = submitted(client, seeded)
    user = client.post(
        "/api/auth/users", {"username": "reviewer", "password": "test-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{setup['project_id']}/members",
        {"user_id": user["id"], "roles": ["reviewer"]},
    )
    client.post("/api/auth/login", {"username": "reviewer", "password": "test-password-1234"})
    corrected = mask.copy()
    corrected.flat[0] = (int(mask.flat[0]) + 1) % 3
    result = complete(http, asset, corrected)
    assert result.status_code == 201 and result.json()["revision"] == 2
    decisions = client.get(f"/api/projects/{setup['project_id']}/decisions")
    assert (
        len(decisions) == 1
        and decisions[0]["revision"] == 2
        and decisions[0]["reviewer_id"] == user["id"]
    )
    binary = http.get(f"/api/annotations/{original['id']}/mask.bin").content
    assert binary == mask.tobytes()
    assert (
        http.get(f"/api/annotations/{result.json()['id']}/mask.bin").content == corrected.tobytes()
    )
    # A stale second reviewer cannot accept over these corrections.
    assert complete(http, asset, mask).status_code == 409
    assert len(client.get(f"/api/assets/{asset['id']}/annotations")) == 2


@pytest.mark.parametrize("role", ["annotator", "outsider"])
def test_only_reviewers_can_accept_corrections(http, client, seeded, role):
    setup, asset, mask, _ = submitted(client, seeded)
    user = client.post("/api/auth/users", {"username": role, "password": "test-password-1234"})
    if role == "annotator":
        client.request(
            "PUT",
            f"/api/projects/{setup['project_id']}/members",
            {"user_id": user["id"], "roles": [role]},
        )
    client.post("/api/auth/login", {"username": role, "password": "test-password-1234"})
    assert complete(http, asset, mask).status_code == 403


def test_incomplete_coverage_never_accepts_or_changes_revision(http, client, seeded):
    setup, asset, mask, _ = submitted(client, seeded)
    assert complete(http, asset, mask, labels=[0, 1]).status_code == 422
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 1
    assert client.get(f"/api/projects/{setup['project_id']}/decisions") == []
