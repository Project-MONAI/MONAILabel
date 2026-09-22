"""Approved data can train without a separate evaluation dataset or invented scores."""

from test_model_splits import add_cases, setup

from monailabel.core.models import Snapshot


def test_training_without_evaluation_and_explicit_grouped_ratio(client, http):
    prefix = setup(client)
    assets = add_cases(client, http, prefix, 0, 5)
    for name, options, expected_train, expected_validation in (
        ("All approved", {}, 5, 0),
        ("With evaluation", {"validation_percentage": 20}, 4, 1),
    ):
        learner = client.post(
            prefix + "/learners",
            {"name": name, "recipe": "pixel-gaussian", "label_ids": [0, 1]},
        )
        job = client.post(prefix + f"/learners/{learner['id']}/train", options)
        result = client.wait(job["id"])
        snapshot = http.app.state.services.store.get(Snapshot, result["snapshot_id"])
        training = {s.asset_id for s in snapshot.samples if s.split == "train"}
        validation = {s.asset_id for s in snapshot.samples if s.split == "validation"}
        assert len(training) == expected_train and len(validation) == expected_validation
        assert not training & validation
        assert training | validation == {a["id"] for a in assets}
        report = client.get(f"/api/jobs/{job['id']}/training-report")
        assert report["error"] is None
        assert report["evaluation_requested"] == bool(expected_validation)
        assert (report["metrics"] is not None) == bool(expected_validation)
        assert client.get(prefix + "/evaluation-sets") == []
        # Omitting the choice retains this model's previous setting.
        again = client.post(prefix + f"/learners/{learner['id']}/train", {})
        client.wait(again["id"])
        assert client.get(f"/api/jobs/{again['id']}/training-report")[
            "evaluation_requested"
        ] == bool(expected_validation)


def test_single_approved_source_can_train_but_cannot_be_its_own_validation(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 1)
    learner = client.post(
        prefix + "/learners",
        {"name": "First model", "recipe": "pixel-gaussian", "label_ids": [0, 1]},
    )
    route = prefix + f"/learners/{learner['id']}/train"
    split = http.post(route, json={"validation_percentage": 20})
    assert split.status_code == 422 and "independent" in split.text
    job = client.post(route, {})
    client.wait(job["id"])
    report = client.get(f"/api/jobs/{job['id']}/training-report")
    assert report["metrics"] is None and report["error"] is None
    assert report["validation_assets"] == []


def test_disabling_evaluation_keeps_prior_heldout_cases_excluded(client, http):
    prefix = setup(client)
    add_cases(client, http, prefix, 0, 5)
    learner = client.post(
        prefix + "/learners",
        {"name": "Stable membership", "recipe": "pixel-gaussian", "label_ids": [0, 1]},
    )
    route = prefix + f"/learners/{learner['id']}/train"
    evaluated = client.wait(client.post(route, {"validation_percentage": 20})["id"])
    store = http.app.state.services.store
    first = store.get(Snapshot, evaluated["snapshot_id"])
    heldout = {s.asset_id for s in first.samples if s.split == "validation"}
    result = client.wait(client.post(route, {"validation_percentage": 0})["id"])
    second = store.get(Snapshot, result["snapshot_id"])
    assert not second.evaluation_requested
    assert len(second.samples) == 4
    assert not heldout & {s.asset_id for s in second.samples}
