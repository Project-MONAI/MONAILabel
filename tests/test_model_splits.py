"""A shared case may train one model while validating another; truth sets never train."""

import io

import numpy as np
import pytest
from PIL import Image

from monailabel.core.evaluation import ModelSplit
from monailabel.core.models import ModelRecord, Snapshot, Split


def add_cases(client, http, prefix, start, stop, *, accepted=True, labels=(0, 1, 2)):
    result = []
    for index in range(start, stop):
        mask = np.zeros((8, 8), np.uint8)
        mask[1:4, 1:4] = 1
        mask[4:7, 4:7] = 2
        image = np.full((8, 8, 3), index, np.uint8)
        image[mask == 1] = 100 + index
        image[mask == 2] = 200 + index
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        response = http.post(
            prefix + "/assets/upload",
            params={"name": f"case-{index}.png", "shared": True},
            content=stream.getvalue(),
        )
        assert response.status_code == 201, response.text
        asset = response.json()
        annotation = client.post(
            f"/api/assets/{asset['id']}/review",
            {
                "base_revision": 0,
                "covered_labels": list(labels),
                "mask": mask.tolist(),
            },
        )
        if accepted:
            client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
        result.append(asset)
    return result


def setup(client):
    project = client.post(
        "/api/projects",
        {
            "name": "Shared data",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Spleen"},
                {"id": 2, "name": "Liver"},
            ],
        },
    )
    return "/api/projects/" + project["id"]


def train(client, prefix, learner, percentage):
    job = client.post(
        prefix + f"/learners/{learner['id']}/train", {"validation_percentage": percentage}
    )
    result = client.wait(job["id"])
    return next(m for m in client.get(prefix + "/models") if m["id"] == result["model_id"]), result


def test_independent_model_splits_grow_and_do_not_reclassify_shared_data(client, http):
    prefix = setup(client)
    original = add_cases(client, http, prefix, 0, 20)
    learners = [
        client.post(
            prefix + "/learners",
            {
                "name": name,
                "recipe": "pixel-gaussian",
                "label_ids": [0, identifier],
            },
        )
        for name, identifier in [("Spleen", 1), ("Liver", 2)]
    ]
    first, report = train(client, prefix, learners[0], 20)
    second, _ = train(client, prefix, learners[1], 25)
    assert len(first["training_assets"]) == 16
    assert len(second["training_assets"]) == 15
    records = client.get(prefix + "/model-splits")
    spleen = next(r for r in records if r["learner_id"] == learners[0]["id"])
    liver = next(r for r in records if r["learner_id"] == learners[1]["id"])
    assert len(spleen["validation_groups"]) == 4
    assert len(liver["validation_groups"]) == 5
    assert set(spleen["training_groups"]) & set(liver["validation_groups"]) or set(
        liver["training_groups"]
    ) & set(spleen["validation_groups"])
    assert all(a["split"] == "pool" for a in client.get(prefix + "/assets"))
    assert client.get(prefix + "/evaluation-sets") == []
    assert client.get("/api/jobs/" + client.get(prefix + "/jobs")[0]["id"])["status"] == "succeeded"
    snapshot = http.app.state.services.store.get(Snapshot, first["snapshot_id"])
    assert snapshot.model_split_id == learners[0]["id"]
    assert snapshot.model_split_version == spleen["version"]
    add_cases(client, http, prefix, 20, 30)
    # Acceptance alone extends the saved sets, before either model runs again.
    before_training = client.get(prefix + "/model-splits")
    assert (
        len(
            next(s for s in before_training if s["learner_id"] == learners[0]["id"])[
                "validation_groups"
            ]
        )
        == 6
    )
    assert (
        len(
            next(s for s in before_training if s["learner_id"] == learners[1]["id"])[
                "validation_groups"
            ]
        )
        == 8
    )
    grown, _ = train(client, prefix, learners[0], 20)
    current = next(
        r for r in client.get(prefix + "/model-splits") if r["learner_id"] == learners[0]["id"]
    )
    assert len(grown["training_assets"]) == 24
    assert len(current["validation_groups"]) == 6
    assert set(spleen["validation_groups"]) <= set(current["validation_groups"])
    assert set(spleen["training_groups"]) <= set(current["training_groups"])
    assert http.app.state.services.store.get(Snapshot, snapshot.id) == snapshot
    liver_now = next(
        r for r in client.get(prefix + "/model-splits") if r["learner_id"] == learners[1]["id"]
    )
    assert len(liver_now["validation_groups"]) == 8
    assert set(liver["validation_groups"]) <= set(liver_now["validation_groups"])
    assert set(liver["training_groups"]) <= set(liver_now["training_groups"])
    # Explicit old snapshots cannot bypass the model's held-out reservation.
    wrong = snapshot.model_copy(
        update={
            "id": "bypass",
            "samples": [s.model_copy(update={"split": Split.TRAIN}) for s in snapshot.samples],
        }
    )
    with http.app.state.services.store.transaction() as session:
        session.insert(wrong)
    response = http.post(
        prefix + "/train",
        json={"snapshot_id": wrong.id, "learner_id": learners[0]["id"], "label_ids": [0, 1]},
    )
    assert response.status_code == 409, response.text
    assert len(original) == 20 and report["training_report_id"]


def test_shared_import_ignores_global_sampling_but_truth_sets_stay_excluded(client, http):
    prefix = setup(client)
    assets = add_cases(client, http, prefix, 0, 2)
    truth = client.post(
        prefix + "/evaluation-sets",
        {
            "name": "External truth",
            "percentage": 100,
            "auto_update": True,
            "asset_ids": [a["id"] for a in assets],
        },
    )
    shared = add_cases(client, http, prefix, 2, 12)
    assert client.get(prefix + "/evaluation-sets")[0] == truth
    learner = client.post(
        prefix + "/learners", {"name": "Spleen", "recipe": "pixel-gaussian", "label_ids": [0, 1]}
    )
    model, _ = train(client, prefix, learner, 20)
    snapshot = http.app.state.services.store.get(Snapshot, model["snapshot_id"])
    assert len(snapshot.samples) == 10
    assert not {a["id"] for a in assets} & {s.asset_id for s in snapshot.samples}
    assert {s.asset_id for s in snapshot.samples} == {a["id"] for a in shared}


def test_model_split_never_turns_training_lineage_or_pending_labels_into_validation(client, http):
    prefix = setup(client)
    assets = add_cases(client, http, prefix, 0, 4)
    learner = client.post(
        prefix + "/learners",
        {"name": "Existing model", "recipe": "pixel-gaussian", "label_ids": [0, 1]},
    )
    with http.app.state.services.store.transaction() as session:
        session.insert(
            ModelRecord(
                project_id=learner["project_id"],
                name="Old checkpoint",
                provider="pixel-gaussian",
                label_ids=[0, 1],
                learner_id=learner["id"],
                training_groups=[a["group_id"] for a in assets],
            )
        )
    response = http.post(
        prefix + f"/learners/{learner['id']}/train", json={"validation_percentage": 20}
    )
    assert response.status_code == 422 and "previously trained" in response.text
    add_cases(client, http, prefix, 4, 6, accepted=False)
    response = http.post(
        prefix + f"/learners/{learner['id']}/train", json={"validation_percentage": 20}
    )
    assert response.status_code == 422
    assert client.get(prefix + "/model-splits") == []
    add_cases(client, http, prefix, 6, 8)
    model, _ = train(client, prefix, learner, 20)
    record = http.app.state.services.store.get(ModelSplit, learner["id"])
    assert not set(record.validation_groups) & {a["group_id"] for a in assets}
    assert len(model["training_assets"]) == 4


@pytest.mark.parametrize("percentage", [0, 51])
def test_validation_percentage_bounds(client, http, percentage):
    prefix = setup(client)
    learner = client.post(
        prefix + "/learners", {"name": "Split", "recipe": "pixel-gaussian", "label_ids": [0, 1]}
    )
    assert (
        http.post(
            prefix + f"/learners/{learner['id']}/train", json={"validation_percentage": percentage}
        ).status_code
        == 422
    )


def test_duplicate_alias_keeps_this_models_validation_membership(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 10)
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Stable patients",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    model, _ = train(client, prefix, learner, 20)
    service = http.app.state.services
    snapshot = service.store.get(Snapshot, model["snapshot_id"])
    validation = next(s for s in snapshot.samples if s.split == Split.VALIDATION)
    asset = client.get("/api/assets/" + validation.asset_id)
    response = http.post(
        prefix + "/assets/upload",
        params={
            "name": "duplicate.png",
            "group_id": "renamed-patient",
            "shared": True,
        },
        content=service.artifacts.read(asset["source_key"]),
    )
    assert response.status_code == 201, response.text
    alias = response.json()
    mask = client.get("/api/annotations/" + asset["annotation_id"] + "/mask")["mask"]
    annotation = client.post(
        f"/api/assets/{alias['id']}/review",
        {
            "base_revision": 0,
            "covered_labels": [0, 1, 2],
            "mask": mask,
        },
    )
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    newer, _ = train(client, prefix, learner, 20)
    current = service.store.get(Snapshot, newer["snapshot_id"])
    pair = [s for s in current.samples if s.image_key == validation.image_key]
    assert len(pair) == 2 and all(s.split == Split.VALIDATION for s in pair)
    assert alias["id"] not in newer["training_assets"]


def test_explicit_truth_set_additions_do_not_sample_a_legacy_percentage(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 10)
    record = client.post(
        prefix + "/evaluation-sets",
        {
            "name": "Existing references",
            "percentage": 20,
            "auto_update": False,
        },
    )
    before = client.get(prefix + "/assets")
    added = add_cases(client, http, prefix, 10, 12)
    updated = client.post(
        prefix + f"/evaluation-sets/{record['id']}/extend",
        {
            "base_version": record["version"],
            "asset_ids": [a["id"] for a in added],
            "include_all": True,
        },
    )
    assert len(updated["member_groups"]) == 4
    assert {a["group_id"] for a in added} <= set(updated["member_groups"])
    assert all(client.get("/api/assets/" + a["id"]) == a for a in before)


def test_existing_fixed_set_trains_shared_images_and_remembers_choice(client, http):
    prefix = setup(client)
    references = add_cases(client, http, prefix, 0, 2)
    fixed = client.post(
        prefix + "/evaluation-sets",
        {
            "name": "External spleen references",
            "percentage": 100,
            "auto_update": False,
            "asset_ids": [a["id"] for a in references],
        },
    )
    training = add_cases(client, http, prefix, 2, 6)
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Spleen fixed evaluation",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    first = client.wait(
        client.post(
            prefix + f"/learners/{learner['id']}/train",
            {
                "evaluation_set_id": fixed["id"],
            },
        )["id"]
    )
    service = http.app.state.services
    frozen = service.store.get(Snapshot, first["snapshot_id"])
    assert {s.asset_id for s in frozen.samples if s.split == Split.TRAIN} == {
        a["id"] for a in training
    }
    assert {s.asset_id for s in frozen.samples if s.split == Split.VALIDATION} == {
        a["id"] for a in references
    }
    saved = next(item for item in client.get(prefix + "/learners") if item["id"] == learner["id"])
    assert saved["evaluation_set_id"] == fixed["id"]
    split = service.store.get(ModelSplit, learner["id"])
    assert split.validation_mode == "fixed" and not split.validation_groups
    added = add_cases(client, http, prefix, 6, 8)
    assert service.store.get(ModelSplit, learner["id"]) == split
    second = client.wait(client.post(prefix + f"/learners/{learner['id']}/train", {})["id"])
    current = service.store.get(Snapshot, second["snapshot_id"])
    assert current.evaluation_version_id == frozen.evaluation_version_id
    assert len([s for s in current.samples if s.split == Split.TRAIN]) == len(training + added)
    assert [s for s in current.samples if s.split == Split.VALIDATION] == [
        s for s in frozen.samples if s.split == Split.VALIDATION
    ]
    assert service.store.get(Snapshot, frozen.id) == frozen
    # Another model can use the fixed references without inheriting a percentage policy.
    other = client.post(
        prefix + "/learners",
        {
            "name": "Other spleen",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    third = client.wait(
        client.post(
            prefix + f"/learners/{other['id']}/train",
            {
                "evaluation_set_id": fixed["id"],
            },
        )["id"]
    )
    assert (
        service.store.get(Snapshot, third["snapshot_id"]).evaluation_version_id
        == frozen.evaluation_version_id
    )


def test_percentage_growth_uses_only_accepted_matching_labels(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 10)
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Spleen only",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    train(client, prefix, learner, 20)
    service = http.app.state.services
    before = service.store.get(ModelSplit, learner["id"])
    add_cases(client, http, prefix, 10, 15, labels=(0, 2))
    assert service.store.get(ModelSplit, learner["id"]) == before
    pending = add_cases(client, http, prefix, 15, 20, accepted=False)
    assert service.store.get(ModelSplit, learner["id"]) == before
    assets = {a["id"]: a for a in client.get(prefix + "/assets")}
    client.post(
        prefix + "/review-decisions",
        {
            "items": [{"annotation_id": assets[a["id"]]["annotation_id"]} for a in pending],
            "verdict": "accepted",
        },
    )
    after = service.store.get(ModelSplit, learner["id"])
    assert len(after.validation_groups) == 3
    assert set(before.validation_groups) <= set(after.validation_groups)
    assert not set(after.validation_groups) & set(before.training_groups)
