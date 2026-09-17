"""Model lifecycle preserves history; run overrides never change recommended defaults."""

import pytest

from monailabel.core.models import Job, JobStatus, Learner, ModelRecord, Snapshot


def setup(client, seeded, recipe="pixel-gaussian"):
    prefix = f"/api/projects/{seeded[0]['project_id']}"
    learner = client.post(
        prefix + "/learners",
        {
            "name": "My specialist",
            "recipe": recipe,
            "label_ids": [0, 1, 2],
        },
    )
    return prefix, learner


def test_setup_rename_delete_and_history(client, http, seeded):
    prefix, learner = setup(client, seeded)
    route = prefix + "/learners/" + learner["id"]
    renamed = client.request("PATCH", route, {"name": "Organ specialist", "base_version": 0})
    assert renamed["config"] == learner["config"]
    assert renamed["version"] == 1
    assert http.patch(route, json={"name": "Stale", "base_version": 0}).status_code == 409
    assert (
        http.patch(route, json={"name": "No", "base_version": 1, "config": {}}).status_code == 422
    )
    assert (
        http.request(
            "DELETE", route, json={"confirmation_name": "Wrong", "base_version": 1}
        ).status_code
        == 422
    )
    client.request("DELETE", route, {"confirmation_name": renamed["name"], "base_version": 1})
    assert client.get(prefix + "/learners") == []
    assert http.post(route + "/train", json={}).status_code == 422
    assert http.app.state.services.store.get(Learner, learner["id"]).archived
    # The name can be reused after deletion.
    client.post(
        prefix + "/learners",
        {"name": renamed["name"], "recipe": "pixel-gaussian", "label_ids": [0, 1, 2]},
    )


def test_model_lifecycle_guards_preserve_checkpoint(client, http, seeded):
    prefix, learner = setup(client, seeded)
    service = http.app.state.services
    model = ModelRecord(
        project_id=seeded[0]["project_id"],
        name="Trained version",
        provider="pixel-gaussian",
        label_ids=[0, 1, 2],
        learner_id=learner["id"],
        state_key="checkpoint",
    )
    with service.store.transaction() as session:
        session.insert(model)
    route = prefix + "/models/" + model.id
    renamed = client.request("PATCH", route, {"name": "Named checkpoint", "base_version": 0})
    assert renamed["state_key"] == model.state_key
    assert renamed["config"] == model.config
    project = client.get(prefix)
    client.request(
        "PUT",
        prefix + "/annotation-model",
        {"model_id": model.id, "base_version": project["version"]},
    )
    body = {"confirmation_name": renamed["name"], "base_version": 1}
    assert http.request("DELETE", route, json=body).status_code == 422
    project = client.get(prefix)
    client.request(
        "PUT",
        prefix + "/annotation-model",
        {"model_id": seeded[0]["baseline_id"], "base_version": project["version"]},
    )
    client.request("DELETE", route, body)
    assert model.id not in {m["id"] for m in client.get(prefix + "/models")}
    assert service.store.get(ModelRecord, model.id).state_key == "checkpoint"
    assert service.store.get(ModelRecord, model.id).archived


def test_setup_controls_enforce_project_role_revision_and_busy_jobs(client, http, seeded):
    prefix, learner = setup(client, seeded)
    route = prefix + "/learners/" + learner["id"]
    other = client.post("/api/projects", {"name": "Other"})
    assert (
        http.patch(
            f"/api/projects/{other['id']}/learners/{learner['id']}",
            json={"name": "No", "base_version": 0},
        ).status_code
        == 422
    )
    job = Job(
        project_id=seeded[0]["project_id"], kind="train", request={}, status=JobStatus.RUNNING
    )
    with http.app.state.services.store.transaction() as session:
        session.insert(job)
    assert (
        http.request(
            "DELETE", route, json={"confirmation_name": learner["name"], "base_version": 0}
        ).status_code
        == 409
    )
    user = client.post(
        "/api/auth/users", {"username": "observer", "password": "test-password-1234"}
    )
    client.request("PUT", prefix + "/members", {"user_id": user["id"], "roles": ["annotator"]})
    http.post("/api/auth/login", json={"username": "observer", "password": "test-password-1234"})
    assert http.patch(route, json={"name": "No", "base_version": 0}).status_code == 403
    assert (
        http.request(
            "DELETE", route, json={"confirmation_name": learner["name"], "base_version": 0}
        ).status_code
        == 403
    )


def test_base_models_cannot_be_modified_or_deleted(client, http, seeded):
    prefix, _ = setup(client, seeded)
    service = http.app.state.services
    base = service.store.get(ModelRecord, seeded[0]["baseline_id"])
    with service.store.transaction() as session:
        session.update(base.model_copy(update={"preset": "test-base"}))
    route = prefix + "/models/" + base.id
    assert http.patch(route, json={"name": "No", "base_version": 0}).status_code == 422
    assert (
        http.request(
            "DELETE", route, json={"confirmation_name": base.name, "base_version": 0}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "config",
    [
        {"epochs": 0},
        {"learning_rate": -1},
        {"batch_size": 0},
        {"batch_size": 33},
        {"weight_decay": -1},
        {"channels": [4, 8, 16, 32]},
        {"label_mapping": {"1": 3}},
    ],
)
def test_invalid_run_overrides_create_no_snapshot_or_job(client, http, seeded, config):
    pytest.importorskip("monailabel.monai")
    prefix, learner = setup(client, seeded, "monai-unet")
    service = http.app.state.services
    before = service.store.list(Snapshot)
    response = http.post(prefix + "/learners/" + learner["id"] + "/train", json={"config": config})
    assert response.status_code == 422
    assert service.store.list(Snapshot) == before
    assert client.get(prefix + "/jobs") == []
    assert service.store.get(Learner, learner["id"]).config == learner["config"]


@pytest.mark.parametrize("recipe", ["monai-unet", "vista3d"])
def test_existing_training_settings_gain_defaults_without_rewriting_setup(client, http, recipe):
    pytest.importorskip("monailabel.monai")
    project = client.post(
        "/api/projects",
        {
            "name": "Existing setup",
            "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Spleen"}],
        },
    )
    pid = project["id"]
    learner = client.post(
        f"/api/projects/{pid}/learners",
        {"name": "Existing", "recipe": recipe, "label_ids": [0, 1]},
    )
    service = http.app.state.services
    record = service.store.get(Learner, learner["id"])
    old_config = {
        key: value
        for key, value in record.config.items()
        if key not in {"batch_size", "weight_decay"}
    }
    legacy = record.model_copy(update={"config": old_config})
    with service.store.transaction() as session:
        session.update(legacy)
    defaults = next(r for r in client.get(f"/api/projects/{pid}/recipes") if r["id"] == recipe)[
        "default_config"
    ]
    assert defaults["batch_size"] == 1
    assert defaults["weight_decay"] == (0.00001 if recipe == "vista3d" else 0)
    from monailabel.core.models import Project

    run = service.learning.run_config(
        legacy, service.store.get(Project, pid), [0, 1], {"batch_size": 3, "weight_decay": 0.01}
    )
    assert run["batch_size"] == 3 and run["weight_decay"] == 0.01
    assert service.store.get(Learner, legacy.id).config == old_config


@pytest.mark.parametrize("origin", ["models", "learners"])
def test_delete_project_model_removes_setup_and_all_versions_atomically(
    client, http, seeded, origin
):
    prefix, learner = setup(client, seeded)
    store = http.app.state.services.store
    versions = [
        ModelRecord(
            project_id=seeded[0]["project_id"],
            name=f"Specialist version {number}",
            provider="pixel-gaussian",
            label_ids=[0, 1, 2],
            learner_id=learner["id"],
            state_key=f"checkpoint-{number}",
        )
        for number in range(2)
    ]
    unrelated = ModelRecord(
        project_id=seeded[0]["project_id"],
        name=learner["name"],
        provider="pixel-gaussian",
        label_ids=[0, 1, 2],
        state_key="unrelated",
    )
    completed = Job(
        project_id=seeded[0]["project_id"],
        kind="train",
        status=JobStatus.SUCCEEDED,
        request={"learner_id": learner["id"]},
        result={"model_id": versions[0].id},
    )
    with store.transaction() as session:
        for record in [*versions, unrelated, completed]:
            session.insert(record)
    expected = {learner["id"]: learner["version"]} | {m.id: m.version for m in versions}
    record = learner if origin == "learners" else versions[0].model_dump()
    client.request(
        "DELETE",
        prefix + f"/{origin}/{record['id']}",
        {
            "confirmation_name": record["name"],
            "base_version": record["version"],
            "scope": "model",
            "related_versions": expected,
        },
    )
    assert client.get(prefix + "/learners") == []
    remaining = {m["id"] for m in client.get(prefix + "/models")}
    assert not {m.id for m in versions} & remaining
    assert unrelated.id in remaining and seeded[0]["baseline_id"] in remaining
    assert store.get(Learner, learner["id"]).archived
    for model in versions:
        saved = store.get(ModelRecord, model.id)
        assert saved.archived and saved.state_key == model.state_key
    assert store.get(Job, completed.id) == completed


@pytest.mark.parametrize(
    "blocker", ["changed_version", "new_version", "default", "dependent_setup"]
)
def test_model_family_delete_guards_leave_every_record_unchanged(client, http, seeded, blocker):
    prefix, learner = setup(client, seeded)
    store = http.app.state.services.store
    model = ModelRecord(
        project_id=seeded[0]["project_id"],
        name="Specialist checkpoint",
        provider="pixel-gaussian",
        label_ids=[0, 1, 2],
        learner_id=learner["id"],
        state_key="checkpoint",
    )
    with store.transaction() as session:
        session.insert(model)
    expected = {learner["id"]: learner["version"], model.id: model.version}
    if blocker == "changed_version":
        client.request(
            "PATCH", prefix + "/models/" + model.id, {"name": "Renamed", "base_version": 0}
        )
    elif blocker == "new_version":
        with store.transaction() as session:
            session.insert(model.model_copy(update={"id": "new-version"}))
    elif blocker == "default":
        project = client.get(prefix)
        client.request(
            "PUT",
            prefix + "/annotation-model",
            {"model_id": model.id, "base_version": project["version"]},
        )
    else:
        with store.transaction() as session:
            session.insert(
                Learner(
                    project_id=seeded[0]["project_id"],
                    protocol_version=1,
                    name="Dependent",
                    recipe="pixel-gaussian",
                    label_ids=[0, 1, 2],
                    initial_model_id=model.id,
                )
            )
    before_models = store.list(ModelRecord, seeded[0]["project_id"])
    before_learners = store.list(Learner, seeded[0]["project_id"])
    response = http.request(
        "DELETE",
        prefix + "/learners/" + learner["id"],
        json={
            "confirmation_name": learner["name"],
            "base_version": learner["version"],
            "scope": "model",
            "related_versions": expected,
        },
    )
    assert response.status_code == (409 if blocker in {"changed_version", "new_version"} else 422)
    assert store.list(ModelRecord, seeded[0]["project_id"]) == before_models
    assert store.list(Learner, seeded[0]["project_id"]) == before_learners


def test_training_rename_updates_matching_version_names_and_preserves_custom_names(
    client, http, seeded
):
    prefix, learner = setup(client, seeded)
    models = [
        ModelRecord(
            project_id=seeded[0]["project_id"],
            name=name,
            provider="pixel-gaussian",
            label_ids=[0, 1, 2],
            learner_id=learner["id"],
            state_key="checkpoint",
        )
        for name in [learner["name"], learner["name"], "Explicit checkpoint name"]
    ]
    store = http.app.state.services.store
    with store.transaction() as session:
        for model in models:
            session.insert(model)
    client.request(
        "PATCH",
        prefix + "/learners/" + learner["id"],
        {"name": "Renamed specialist", "base_version": 0},
    )
    for model in models[:2]:
        saved = store.get(ModelRecord, model.id)
        assert saved.name == "Renamed specialist" and saved.version == 1
        assert saved.state_key == model.state_key
    assert store.get(ModelRecord, models[2].id) == models[2]
