"""Evaluation reservations survive growth, history reuse and management changes."""

import io

import numpy as np
from PIL import Image

from monailabel.core.evaluation import EvaluationSetVersion
from monailabel.core.models import Job, Snapshot


def project(client):
    return client.post(
        "/api/projects",
        {
            "name": "Evaluation experiment",
            "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Spleen"}],
        },
    )["id"]


def upload(http, pid, number, **params):
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:] = (number, 15, 25)
    pixels[2:5, 2:5] = (200, number, 20)
    content = io.BytesIO()
    Image.fromarray(pixels).save(content, format="PNG")
    response = http.post(
        f"/api/projects/{pid}/assets/upload",
        params={"name": f"{number}.png", "group_id": f"patient-{number}"} | params,
        content=content.getvalue(),
    )
    return response


def submit(client, asset):
    return client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": asset["revision"],
            "covered_labels": [0, 1],
            "mask": [[int(2 <= x < 5 and 2 <= y < 5) for x in range(8)] for y in range(8)],
        },
    )


def accept_all(client, pid):
    for asset in client.get(f"/api/projects/{pid}/assets"):
        annotation = submit(client, asset)
        client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})


def test_percentage_grows_without_reshuffling_and_retries_do_not_add_members(client, http):
    pid = project(client)
    for i in range(10):
        assert upload(http, pid, i).status_code == 201
    prefix = f"/api/projects/{pid}/evaluation-sets"
    record = client.post(prefix, {"name": "Spleen evaluation", "percentage": 20})
    original = set(record["member_groups"])
    assert len(original) == 2
    for i in range(10, 15):
        assert upload(http, pid, i).status_code == 201
    record = client.get(prefix)[0]
    assert len(record["member_groups"]) == 3
    assert original <= set(record["member_groups"])
    assert len(record["cohort_groups"]) == 15
    for i in range(15):
        assert upload(http, pid, i).status_code == 201
    assert client.get(prefix)[0] == record
    assets = client.get(f"/api/projects/{pid}/assets")
    assert len(assets) == 15
    assert sum(a["split"] == "validation" for a in assets) == 3
    # Freeze future growth while preserving reservations and current members.
    frozen = client.request(
        "PATCH",
        prefix + "/" + record["id"],
        {"base_version": record["version"], "auto_update": False, "name": "Stable spleen"},
    )
    assert upload(http, pid, 99).json()["split"] == "pool"
    assert client.get(prefix)[0] == frozen
    assert (
        http.patch(
            prefix + "/" + record["id"], json={"base_version": 0, "name": "Stale"}
        ).status_code
        == 409
    )
    assert http.post(prefix, json={"name": "stable SPLEEN"}).status_code == 409


def test_versions_preserve_references_and_train_compare_reuse_same_version(client, http):
    pid = project(client)
    prefix = f"/api/projects/{pid}"
    for i in range(10):
        upload(http, pid, i).raise_for_status()
    record = client.post(
        prefix + "/evaluation-sets", {"name": "Shared evaluation", "percentage": 20}
    )
    route = prefix + "/evaluation-sets/" + record["id"] + "/versions"
    assert (
        http.post(route, json={"base_version": record["version"], "label_ids": [0, 1]}).status_code
        == 422
    )
    accept_all(client, pid)
    version = client.post(route, {"base_version": record["version"], "label_ids": [0, 1]})
    latest = client.get(prefix + "/evaluation-sets")[0]
    assert client.post(route, {"base_version": latest["version"], "label_ids": [0, 1]}) == version
    learner = client.post(
        prefix + "/learners", {"name": "Student", "recipe": "pixel-gaussian", "label_ids": [0, 1]}
    )
    models = []
    for _ in range(2):
        result = client.wait(client.post(prefix + f"/learners/{learner['id']}/train", {})["id"])
        models.append(result["model_id"])
        snapshot = http.app.state.services.store.get(Snapshot, result["snapshot_id"])
        assert snapshot.evaluation_version_id == version["id"]
        assert [
            s.model_dump(mode="json") for s in snapshot.samples if s.split == "validation"
        ] == version["samples"]
    evaluation = client.wait(
        client.post(
            prefix + "/evaluate",
            {
                "evaluation_version_id": version["id"],
                "candidate_id": models[1],
                "baseline_id": models[0],
            },
        )["id"]
    )
    result = client.get("/api/evaluations/" + evaluation["evaluation_id"])
    assert result["evaluation_version_id"] == version["id"] and result["snapshot_id"] is None
    assert result["candidate"]["mean_dice"] > 0.99
    # An edited review can be reset/reaccepted without rewriting the frozen reference.
    asset = client.get("/api/assets/" + version["samples"][0]["asset_id"])
    annotation = submit(client, asset)
    client.post("/api/annotations/" + annotation["id"] + "/decision", {"verdict": "accepted"})
    next_version = client.post(route, {"base_version": latest["version"], "label_ids": [0, 1]})
    assert next_version["number"] == 2
    saved = http.app.state.services.store.get(EvaluationSetVersion, version["id"])
    assert saved.samples[0].revision == version["samples"][0]["revision"]
    assert len(client.get(prefix + "/evaluation-set-versions")) == 2
    current = client.get(prefix + "/evaluation-sets")[0]
    assert (
        http.request(
            "DELETE",
            prefix + "/evaluation-sets/" + record["id"],
            json={"base_version": current["version"]},
        ).status_code
        == 409
    )
    archived = client.request(
        "PATCH",
        prefix + "/evaluation-sets/" + record["id"],
        {"base_version": current["version"], "archived": True},
    )
    # Existing immutable versions remain valid for comparisons after archive.
    assert client.wait(
        client.post(
            prefix + "/evaluate",
            {
                "evaluation_version_id": version["id"],
                "candidate_id": models[0],
                "baseline_id": models[1],
            },
        )["id"]
    )["evaluation_id"]
    restored = client.request(
        "PATCH",
        prefix + "/evaluation-sets/" + record["id"],
        {"base_version": archived["version"], "archived": False},
    )
    assert restored["latest_version_id"] == next_version["id"]


def test_reservations_block_older_training_snapshots_and_duplicate_images(client, http):
    pid = project(client)
    prefix = f"/api/projects/{pid}"
    for i in range(5):
        upload(http, pid, i, split="train").raise_for_status()
    accept_all(client, pid)
    snapshot = client.post(prefix + "/snapshots")
    record = client.post(prefix + "/evaluation-sets", {"name": "Reserved", "percentage": 20})
    assert http.post(prefix + "/train", json={"snapshot_id": snapshot["id"]}).status_code == 409
    reserved = next(a for a in client.get(prefix + "/assets") if a["split"] == "validation")
    number = int(reserved["name"].split(".")[0])
    assert upload(http, pid, number, group_id="renamed-patient", split="train").status_code == 409
    duplicate = upload(http, pid, number, group_id="renamed-patient", name="renamed.png").json()
    assert duplicate["split"] == "validation"
    # No reference versions: delete the name, preserving evaluation-only data.
    current = client.get(prefix + "/evaluation-sets")[0]
    client.request(
        "DELETE", prefix + "/evaluation-sets/" + record["id"], {"base_version": current["version"]}
    )
    assert client.get(prefix + "/evaluation-sets") == []
    assert http.post("/api/assets/" + reserved["id"] + "/assign-train").status_code == 409
    assert (
        http.request(
            "DELETE", prefix + "/assets", json={"asset_ids": [duplicate["id"]]}
        ).status_code
        == 409
    )
    assert upload(http, pid, number, group_id="third-name", split="train").status_code == 409


def test_active_and_historical_training_groups_are_not_new_holdouts(client, http):
    pid = project(client)
    prefix = f"/api/projects/{pid}"
    for i in range(4):
        upload(http, pid, i, split="train").raise_for_status()
    accept_all(client, pid)
    snapshot = client.post(prefix + "/snapshots")
    store = http.app.state.services.store
    with store.transaction() as session:
        session.insert(Job(project_id=pid, kind="train", request={"snapshot_id": snapshot["id"]}))
    record = client.post(
        prefix + "/evaluation-sets", {"name": "Later evaluation", "percentage": 50}
    )
    assert record["member_groups"] == []
    for i in range(4, 8):
        upload(http, pid, i).raise_for_status()
    record = client.get(prefix + "/evaluation-sets")[0]
    assert set(record["member_groups"]) == {f"patient-{i}" for i in range(4, 8)}


def test_permissions_and_project_boundaries(client, http):
    first, second = project(client), project(client)
    sample = upload(http, first, 1).json()
    prefix = f"/api/projects/{first}/evaluation-sets"
    record = client.post(prefix, {"name": "One"})
    assert (
        http.patch(
            f"/api/projects/{second}/evaluation-sets/{record['id']}",
            json={"base_version": record["version"], "name": "Wrong"},
        ).status_code
        == 422
    )
    assert (
        http.post(
            f"/api/projects/{second}/evaluation-sets",
            json={"name": "Wrong", "asset_ids": [sample["id"]]},
        ).status_code
        == 422
    )
    user = client.post("/api/auth/users", {"username": "reviewer", "password": "reviewer-password"})
    client.request(
        "PUT", f"/api/projects/{first}/members", {"user_id": user["id"], "roles": ["reviewer"]}
    )
    client.post("/api/auth/login", {"username": "reviewer", "password": "reviewer-password"})
    assert http.post(prefix, json={"name": "Forbidden"}).status_code == 403
    assert http.get(prefix).status_code == 200


def test_start_training_prepares_and_reuses_references_without_manual_save(client, http):
    pid = client.post(
        "/api/projects",
        {
            "name": "Automatic evaluation references",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Spleen"},
                {"id": 2, "name": "Liver"},
            ],
        },
    )["id"]
    prefix = f"/api/projects/{pid}"
    for i in range(5):
        upload(http, pid, i).raise_for_status()
    record = client.post(prefix + "/evaluation-sets", {"name": "Reserved", "percentage": 20})
    accept_all(client, pid)  # Only spleen is covered; liver is not a training target.
    learner = client.post(
        prefix + "/learners", {"name": "Student", "recipe": "pixel-gaussian", "label_ids": [0, 1]}
    )
    route = prefix + f"/learners/{learner['id']}/train"
    assert client.get(prefix + "/evaluation-set-versions") == []
    first = client.wait(client.post(route, {})["id"])
    version = client.get(prefix + "/evaluation-set-versions")[0]
    assert [label["id"] for label in version["labels"]] == [0, 1]
    assert version["evaluation_set_id"] == record["id"] and version["published_by"] != "system"
    assert len(version["samples"]) == 1
    snapshot = http.app.state.services.store.get(Snapshot, first["snapshot_id"])
    assert snapshot.evaluation_version_id == version["id"]
    assert not any(
        s.group_id in record["member_groups"] for s in snapshot.samples if s.split == "train"
    )
    # Current annotations may change; the set's saved default remains a fixed reference.
    reserved = client.get("/api/assets/" + version["samples"][0]["asset_id"])
    changed = submit(client, reserved)
    client.post("/api/annotations/" + changed["id"] + "/decision", {"verdict": "pending"})
    second = client.wait(client.post(route, {"evaluation_set_id": record["id"]})["id"])
    snapshot = http.app.state.services.store.get(Snapshot, second["snapshot_id"])
    assert snapshot.evaluation_version_id == version["id"]
    assert client.get(prefix + "/evaluation-set-versions") == [version]


def test_automatic_references_require_review_and_an_unambiguous_project_set(client, http):
    pid = project(client)
    prefix = f"/api/projects/{pid}"
    for i in range(5):
        upload(http, pid, i).raise_for_status()
    first = client.post(prefix + "/evaluation-sets", {"name": "First", "percentage": 20})
    learner = client.post(
        prefix + "/learners", {"name": "Student", "recipe": "pixel-gaussian", "label_ids": [0, 1]}
    )
    route = prefix + f"/learners/{learner['id']}/train"
    assert http.post(route, json={}).status_code == 422
    assert client.get(prefix + "/evaluation-set-versions") == []
    assert client.get(prefix + "/jobs") == []
    second = client.post(prefix + "/evaluation-sets", {"name": "Second", "percentage": 20})
    accept_all(client, pid)
    response = http.post(route, json={})
    assert response.status_code == 422 and "Choose an evaluation set" in response.text
    other = project(client)
    foreign = client.post(f"/api/projects/{other}/evaluation-sets", {"name": "Foreign"})
    assert http.post(route, json={"evaluation_set_id": foreign["id"]}).status_code == 422
    client.request(
        "PATCH",
        prefix + "/evaluation-sets/" + first["id"],
        {"base_version": first["version"], "archived": True},
    )
    assert http.post(route, json={"evaluation_set_id": first["id"]}).status_code == 409
    assert client.get(prefix + "/evaluation-set-versions") == []
    result = client.wait(client.post(route, {"evaluation_set_id": second["id"]})["id"])
    version = client.get(prefix + "/evaluation-set-versions")[0]
    assert version["evaluation_set_id"] == second["id"]
    assert (
        http.app.state.services.store.get(Snapshot, result["snapshot_id"]).evaluation_version_id
        == version["id"]
    )
    assert (
        http.post(
            route, json={"evaluation_set_id": second["id"], "evaluation_version_id": version["id"]}
        ).status_code
        == 422
    )
