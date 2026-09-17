"""Run filters narrow training without changing held-out evaluation membership."""

import pytest
from test_model_splits import add_cases, setup, train

from monailabel.core.models import (
    Asset,
    DicomSeries,
    Job,
    Sample,
    Snapshot,
    Split,
    TrainingSampleFilter,
)
from monailabel.server.training_samples import filter_samples


@pytest.mark.parametrize("fixed", [False, True])
def test_training_filters_preserve_evaluation_and_snapshot_history(client, http, fixed):
    prefix = setup(client)
    assets = add_cases(client, http, prefix, 0, 12)
    request = {"validation_percentage": 20}
    if fixed:
        reference = client.post(
            prefix + "/evaluation-sets",
            {
                "name": "External",
                "percentage": 100,
                "auto_update": False,
                "asset_ids": [a["id"] for a in assets[:2]],
            },
        )
        request = {"evaluation_set_id": reference["id"]}
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Spleen",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    route = prefix + f"/learners/{learner['id']}/train"
    first = client.wait(client.post(route, request)["id"])
    store = http.app.state.services.store
    original = store.get(Snapshot, first["snapshot_id"])
    available = [s.asset_id for s in original.samples if s.split == Split.TRAIN]
    # Import provenance, not filenames, determines the dataset/source filter.
    with store.transaction() as session:
        session.insert(
            Job(
                project_id=prefix.split("/")[-1],
                kind="dataset_import",
                request={"template_id": "Task09_Spleen"},
                result={"asset_ids": available[:4]},
            )
        )
    filters = {"source_ids": ["template:Task09_Spleen"], "asset_ids": available[1:], "limit": 2}
    second = client.wait(client.post(route, {**request, "sample_filter": filters})["id"])
    filtered = store.get(Snapshot, second["snapshot_id"])
    actual = {s.asset_id for s in filtered.samples if s.split == Split.TRAIN}
    assert len(actual) == 2 and actual <= set(available[1:4])
    assert filtered.sample_filter.model_dump() == filters
    assert [s for s in filtered.samples if s.split == Split.VALIDATION] == [
        s for s in original.samples if s.split == Split.VALIDATION
    ]
    assert store.get(Snapshot, original.id) == original
    sources = client.get(prefix + "/training-sources")
    assert (
        next(s for s in sources if s["id"] == "template:Task09_Spleen")["asset_ids"]
        == available[:4]
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"source_ids": ["missing"]},
        {"asset_ids": ["another-project"]},
        {"limit": 0},
        {"asset_ids": []},
    ],
)
def test_invalid_filters_do_not_publish_training_jobs_or_splits(client, http, filters):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 5)
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Spleen",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    response = http.post(
        prefix + f"/learners/{learner['id']}/train",
        json={"validation_percentage": 20, "sample_filter": filters},
    )
    assert response.status_code in {409, 422}, response.text
    assert client.get(prefix + "/model-splits") == []
    assert client.get(prefix + "/jobs") == []


def test_selecting_only_evaluation_images_cannot_train(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 5)
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Spleen",
            "recipe": "pixel-gaussian",
            "label_ids": [0, 1],
        },
    )
    model, _ = train(client, prefix, learner, 20)
    snapshot = http.app.state.services.store.get(Snapshot, model["snapshot_id"])
    held_out = [s.asset_id for s in snapshot.samples if s.split == Split.VALIDATION]
    response = http.post(
        prefix + f"/learners/{learner['id']}/train", json={"sample_filter": {"asset_ids": held_out}}
    )
    assert response.status_code in {409, 422} and "No eligible training" in response.text, (
        response.text
    )
    assert len(client.get(prefix + "/jobs")) == 1


def test_limit_keeps_related_samples_together_and_ignores_viewing_copies(client, http):
    prefix = setup(client)
    assets = add_cases(client, http, prefix, 0, 6)
    project_id = prefix.split("/")[-1]
    store = http.app.state.services.store
    with store.transaction() as session:
        samples = []
        for index, item in enumerate(assets):
            asset = session.get(Asset, item["id"])
            asset = asset.model_copy(update={"group_id": f"patient-{index // 2}"})
            session.update(asset)
            samples.append(
                Sample(
                    asset_id=asset.id,
                    image_key=asset.image_key,
                    mask_key="unused",
                    revision=1,
                    group_id=asset.group_id,
                    split=Split.TRAIN if index < 4 else Split.VALIDATION,
                )
            )
        session.insert(
            DicomSeries(
                project_id=project_id,
                asset_id=assets[0]["id"],
                study_uid="1",
                series_uid="2",
                frame_of_reference_uid="3",
                instance_uids=[],
                source_keys=[],
                derived_from_nifti=True,
            )
        )
        filtered = filter_samples(session, project_id, samples, TrainingSampleFilter(limit=3))
        assert filtered == filter_samples(
            session, project_id, samples, TrainingSampleFilter(limit=3)
        )
        training = [s for s in filtered if s.split == Split.TRAIN]
        assert len(training) == 2 and len({s.group_id for s in training}) == 1
        assert [s for s in filtered if s.split == Split.VALIDATION] == samples[-2:]
    assert client.get(prefix + "/training-sources") == [
        {
            "id": "uploads",
            "name": "Uploaded files",
            "asset_ids": [a["id"] for a in assets],
        }
    ]
