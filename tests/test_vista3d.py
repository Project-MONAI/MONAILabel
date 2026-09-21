import gzip

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord, Snapshot

pytestmark = pytest.mark.filterwarnings(
    "ignore:The cuda.cudart module is deprecated.*:FutureWarning"
)


@pytest.mark.parametrize("hosted_key", [False, True])
def test_presets_default_to_vista_without_overriding_user_choices(
    client, http, monkeypatch, hosted_key
):
    if hosted_key:
        monkeypatch.setenv("NV_INFERENCE_API_KEY", "test-key-never-sent")
    else:
        monkeypatch.delenv("NV_INFERENCE_API_KEY", raising=False)
    service = http.app.state.services
    service.presets.enabled = True
    project = client.post(
        "/api/projects",
        {
            "name": "Radiology",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Spleen", "color": "#00ff00"},
                {"id": 2, "name": "Custom structure", "color": "#ff0000"},
            ],
        },
    )
    prefix = f"/api/projects/{project['id']}"
    first = client.get(prefix + "/models")
    assert len(first) == 6
    base = next(m for m in first if m["preset"] == "vista3d")
    assert base["read_only"] and base["state_key"] is None and base["label_ids"] == [0]
    sol = next(m for m in first if m["preset"] == "nvidia-sol")
    claude = next(m for m in first if m["preset"] == "nvidia-claude-opus-5")
    assert claude["name"] == "Claude Opus 5"
    assert claude["config"]["model"] == "azure/anthropic/claude-opus-5"
    assert claude["config"]["token_env"] == "NV_INFERENCE_API_KEY"
    assert "reasoning_effort" not in claude["config"]
    assert project["annotation_model_id"] == base["id"]
    assert project["defaults"] == {"1": base["id"]}
    assert "test-key-never-sent" not in str(first)
    service.presets.ensure(project["id"])
    assert client.get(prefix + "/models") == first
    assert client.get(prefix) == project
    selected = client.request(
        "PUT",
        prefix + "/annotation-model",
        {"model_id": sol["id"], "base_version": project["version"]},
    )
    service.presets.ensure(project["id"])
    assert client.get(prefix) == selected


def test_preset_title_update_preserves_configuration_and_custom_names(client, http):
    service = http.app.state.services
    service.presets.enabled = True
    project = client.post("/api/projects", {"name": "Existing project"})
    old_names = {
        "vista3d": "VISTA3D · CT foundation · read-only",
        "nvidia-sol": "GPT-5.6 Sol · NVIDIA gateway",
        "nvidia-astra": "GPT-6 Astra · NVIDIA gateway (paid)",
    }
    before = service.store.list(ModelRecord, project["id"])
    assert {m.name for m in before} == {
        "VISTA3D",
        "GPT-5.6 Sol",
        "GPT-6 Astra",
        "Claude Opus 5",
        "SAM 2.1",
        "MedSAM2",
    }
    with service.store.transaction() as session:
        for model in before:
            session.update(
                model.model_copy(update={"name": old_names.get(model.preset, model.name)})
            )
    service.presets.ensure(project["id"])
    assert service.store.list(ModelRecord, project["id"]) == before
    assert client.get(f"/api/projects/{project['id']}") == project
    sol = next(m for m in before if m.preset == "nvidia-sol")
    custom = sol.model_copy(update={"name": "Our annotation model"})
    with service.store.transaction() as session:
        session.update(custom)
    service.presets.ensure(project["id"])
    assert service.store.get(ModelRecord, sol.id) == custom


def test_vista_default_vocabulary_is_available_without_framework_imports(client):
    from monailabel.core.models import Label
    from monailabel.providers.vista3d import mapping, targets

    project = client.post("/api/projects", {"name": "Vocabulary"})
    recipe = next(
        r for r in client.get(f"/api/projects/{project['id']}/recipes") if r["id"] == "vista3d"
    )
    assert len(recipe["supported_targets"]) == 117
    assert recipe["target_class_ids"] == targets()
    assert recipe["target_class_ids"]["liver"] == 1
    assert recipe["target_class_ids"]["spleen"] == 3
    assert set(targets().values()) == set(range(1, 133)) - {
        2,
        16,
        18,
        20,
        21,
        23,
        24,
        25,
        26,
        27,
        128,
        129,
        130,
        131,
        132,
    }
    assert mapping([Label(id=1, name="Vertebrae L5", color="#ffffff")]) == {1: 33}
    assert "c6dbe159632a4767696e09f91d74d729b82e73e6" in recipe["documentation_url"]


@pytest.fixture
def vista_project(client, http):
    pytest.importorskip("monailabel.monai")
    http.app.state.services.presets.enabled = True
    project = client.post(
        "/api/projects",
        {
            "name": "Organ learning",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 5, "name": "Spleen", "color": "#00ff00"},
                {"id": 9, "name": "Liver", "color": "#ff0000"},
                {"id": 12, "name": "Other", "color": "#0000ff"},
            ],
        },
    )
    prefix = f"/api/projects/{project['id']}"
    base = next(m for m in client.get(prefix + "/models") if m["preset"] == "vista3d")
    return project, base


def reviewed_volumes(client, http, project, *, shared=False):
    mask = np.zeros((16, 16, 16), np.uint8)
    mask[2:6, 3:7, 4:8] = 5
    mask[9:13, 8:12, 7:11] = 9
    mask[0, 0, 0] = 12
    affine = np.diag([-1.5, 1.5, 3.0, 1.0])
    assets, originals = [], []
    for group, split in [
        ("patient-a", "train"),
        ("patient-b", "validation"),
        ("patient-c", "train"),
    ]:
        # Independent patient fixtures must not be identical decoded images.
        source = nib.Nifti1Image(mask.astype(np.float32) + len(assets), affine)
        response = http.post(
            f"/api/projects/{project['id']}/assets/upload",
            params={
                "name": group + ".nii.gz",
                "group_id": group,
                "split": "pool" if shared else split,
                "shared": shared,
            },
            content=gzip.compress(source.to_bytes()),
        )
        assert response.status_code == 201
        asset = response.json()
        annotation = client.post(
            f"/api/assets/{asset['id']}/review",
            {"base_revision": 0, "mask": mask.tolist(), "covered_labels": [0, 5, 9, 12]},
        )
        if group != "patient-c":
            client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
        assets.append(asset)
        originals.append(annotation)
    return assets, originals


@pytest.mark.parametrize("reverse", [False, True])
def test_compare_inherited_model_and_base_resolves_trained_targets(
    client, http, vista_project, monkeypatch, reverse
):
    project, base = vista_project
    reviewed_volumes(client, http, project)
    service = http.app.state.services
    prefix = f"/api/projects/{project['id']}"
    child = ModelRecord(
        project_id=project["id"],
        name="Spleen specialist",
        provider="vista3d",
        label_ids=[0, 5],
        inherit_targets=True,
        parent_id=base["id"],
        training_groups=["patient-a"],
    )
    with service.store.transaction() as session:
        session.insert(child)
    calls = []

    def predict(project, model, image, prompt, affine):
        calls.append((model.id, model.label_ids))
        # Independent deterministic runtime fixture; no foundation weights loaded.
        mask = np.rint(image[..., 0] - 1).astype(np.uint8)
        return np.where(np.isin(mask, model.label_ids), mask, 0).astype(np.uint8)

    monkeypatch.setattr(service.models, "predict", predict)
    pair = [child.id, base["id"]]
    if reverse:
        pair.reverse()
    body = {"candidate_id": pair[0], "baseline_id": pair[1]}
    result = client.wait(client.post(prefix + "/evaluate", body)["id"])
    comparison = client.get("/api/evaluations/" + result["evaluation_id"])
    assert comparison["candidate"]["per_class"] == {"5": 1.0}
    assert comparison["baseline"]["per_class"] == {"5": 1.0}
    assert calls == [(pair[0], [0, 5]), (pair[1], [0, 5])]
    # Explicit inherited anatomy is allowed; unsupported project labels are not.
    result = client.wait(client.post(prefix + "/evaluate", body | {"label_ids": [9]})["id"])
    comparison = client.get("/api/evaluations/" + result["evaluation_id"])
    assert comparison["candidate"]["per_class"] == {"9": 1.0}
    before = len(client.get(prefix + "/jobs"))
    response = http.post(prefix + "/evaluate", json=body | {"label_ids": [12]})
    assert response.status_code == 422
    assert len(client.get(prefix + "/jobs")) == before
    assert service.store.get(ModelRecord, child.id) == child
    assert service.store.get(ModelRecord, base["id"]).model_dump(mode="json") == base


def test_missing_reviewed_organs_prepares_setup_without_launching_training(
    client, http, vista_project
):
    project, base = vista_project
    prefix = f"/api/projects/{project['id']}"
    reply = client.post(
        prefix + "/assistant",
        {
            "message": "i want to finetune a new model for spleen and liver "
            "from the annotated dataset"
        },
    )
    assert reply["job_id"] is None and "review" in reply["message"]
    learner = client.get(prefix + "/learners")[0]
    assert learner["initial_model_id"] == base["id"]
    assert learner["label_ids"] == [0, 5, 9]
    assert learner["config"]["label_mapping"] == {"5": 3, "9": 1}
    assert client.get(prefix + "/jobs") == []
    assert next(m for m in client.get(prefix + "/models") if m["id"] == base["id"]) == base


def test_training_filters_labels_preserves_geometry_and_forks_base(
    client, http, vista_project, monkeypatch
):
    project, base = vista_project
    assets, annotations = reviewed_volumes(client, http, project, shared=True)
    service = http.app.state.services
    calls = []

    class Trainer:
        def train_volumes(self, samples, label_ids, mode, parent_state, progress):
            calls.extend(samples)
            assert label_ids == [0, 5, 9] and mode == "fine_tune"
            assert parent_state == {"format": "vista3d-base-v1"}
            assert len(samples) == 1  # No held-out or unreviewed source enters optimization.
            assert samples[0].volume.affine == assets[0]["affine"]
            assert set(np.unique(samples[0].mask)) == {0, 5, 9}
            progress(1)
            return {"format": "fixture-checkpoint"}

    monkeypatch.setattr(service.learning.recipes, "trainer", lambda recipe, config: Trainer())
    prefix = f"/api/projects/{project['id']}"
    reply = client.post(
        prefix + "/assistant",
        {
            "message": "i want to train/finetune a new model for spleen and liver "
            "from the annotated dataset"
        },
    )
    result = client.wait(reply["job_id"])
    child = next(m for m in client.get(prefix + "/models") if m["id"] == result["model_id"])
    assert not child["read_only"] and child["parent_id"] == base["id"]
    assert child["training_groups"] in (["patient-a"], ["patient-b"])
    snapshot = service.store.get(Snapshot, result["snapshot_id"])
    assert [label.id for label in snapshot.labels] == [0, 5, 9]
    assert {s.group_id for s in snapshot.samples} == {"patient-a", "patient-b"}
    assert snapshot.samples[0].affine == assets[0]["affine"]
    assert len(calls) == 1
    original = np.frombuffer(
        http.get(f"/api/annotations/{annotations[0]['id']}/mask.bin").content, np.uint8
    )
    assert 12 in original
    assert service.store.get(ModelRecord, base["id"]).model_dump(mode="json") == base
    job_count = len(client.get(prefix + "/jobs"))
    response = http.post(
        prefix + "/learners/" + child["learner_id"] + "/train",
        json={"mode": "continue", "parent_model_id": base["id"]},
    )
    assert response.status_code == 422
    assert len(client.get(prefix + "/jobs")) == job_count


def test_inherited_setup_chooses_organs_per_run_and_keeps_base_vocabulary(
    client, http, vista_project, monkeypatch
):
    project, base = vista_project
    assets, _ = reviewed_volumes(client, http, project, shared=True)
    service = http.app.state.services
    prefix = f"/api/projects/{project['id']}"
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="inherit",
                    name="create_learner",
                    arguments={"recipe": "vista3d", "initialization": "fine_tune"},
                )
            ],
        )
    ]
    reply = client.post(prefix + "/assistant", {"message": "Create a VISTA3D project model"})
    learner = client.get(prefix + "/learners")[0]
    assert reply["job_id"] is None and learner["inherit_targets"]
    assert learner["label_ids"] == [0] and learner["config"]["label_mapping"] == {}
    assert client.get(prefix)["labels"] == project["labels"]
    path = prefix + "/learners/" + learner["id"] + "/train"
    response = http.post(path, json={"mode": "fine_tune", "parent_model_id": base["id"]})
    assert response.status_code == 422 and "Choose annotated organs" in response.json()["detail"]
    assert client.get(prefix + "/jobs") == []
    configs = []

    class Trainer:
        def train_volumes(self, samples, label_ids, mode, parent_state, progress):
            assert len(samples) == 1
            assert set(np.unique(samples[0].mask)) == set(label_ids)
            assert samples[0].volume.affine == assets[0]["affine"]
            progress(1)
            return {"format": "fixture-checkpoint"}

    def trainer(recipe, config):
        configs.append(config)
        return Trainer()

    monkeypatch.setattr(service.learning.recipes, "trainer", trainer)
    # Explicit fine-tune can use the learner's base even with a hosted annotator selected.
    sol = next(m for m in client.get(prefix + "/models") if m["preset"] == "nvidia-sol")
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="spleen-run",
                    name="start_training",
                    arguments={"mode": "fine_tune", "targets": ["spleen"]},
                )
            ],
        )
    ]
    reply = client.post(
        prefix + "/assistant",
        {
            "message": "Fine-tune this model for spleen",
            "context": {"learner_id": learner["id"], "model_id": sol["id"]},
        },
    )
    result = client.wait(reply["job_id"])
    child = service.store.get(ModelRecord, result["model_id"])
    assert child.inherit_targets and child.label_ids == [0, 5]
    service.models.validate_targets(child, ["liver", "spleen"])
    with pytest.raises(DomainError, match="does not support"):
        service.models.validate_targets(child, ["Other"])
    second = client.wait(
        client.post(
            path,
            {
                "mode": "fine_tune",
                "parent_model_id": child.id,
                "label_ids": [0, 9],
            },
        )["id"]
    )
    assert configs[0]["label_mapping"] == {"5": 3}
    assert configs[1]["label_mapping"] == {"9": 1}
    assert service.store.get(ModelRecord, second["model_id"]).parent_id == child.id
    assert [label.id for label in service.store.get(Snapshot, second["snapshot_id"]).labels] == [
        0,
        9,
    ]
    count = len(client.get(prefix + "/jobs"))
    for labels, mode in [([0, 9], "continue"), ([0, 12], "fine_tune")]:
        response = http.post(
            path, json={"mode": mode, "parent_model_id": child.id, "label_ids": labels}
        )
        assert response.status_code == 422
    assert len(client.get(prefix + "/jobs")) == count
    assert client.get(prefix + "/learners")[0] == learner
    assert service.store.get(ModelRecord, base["id"]).model_dump(mode="json") == base


def test_vista_geometry_and_project_id_mapping_with_native_monai_transforms(tmp_path, monkeypatch):
    pytest.importorskip("monailabel.monai")
    import torch

    from monailabel.core.models import Label
    from monailabel.core.ports import Volume
    from monailabel.monai import vista_runtime as runtime
    from monailabel.server.storage import Artifacts

    class Network(torch.nn.Module):
        def forward(self, x, class_vector, transpose):
            assert class_vector.tolist() == [[3], [1]]
            return torch.cat([(x - 0.65) * 20, (0.35 - x) * 20], dim=1)

    monkeypatch.setattr(runtime, "vista3d132", Network)
    monkeypatch.setattr(runtime, "load_base", lambda net: None)
    image = np.zeros((32, 24, 16, 1), np.float32)
    image[3:11, 4:12, 2:9] = 500
    image[19:27, 14:20, 4:11] = -500
    expected = np.where(image[..., 0] > 0, 5, np.where(image[..., 0] < 0, 9, 0)).astype(np.uint8)
    model = ModelRecord(
        name="VISTA",
        provider="vista3d",
        read_only=True,
        label_ids=[0, 5, 9],
        config={"device": "cpu", "patch_size": 32},
    )
    labels = [
        Label(id=5, name="Spleen", color="#00ff00"),
        Label(id=9, name="Liver", color="#ff0000"),
    ]
    volume = Volume(image, np.diag([-1.5, 1.5, 3.0, 1.0]).tolist())
    actual = (
        runtime.VistaSegmenter({}, Artifacts(tmp_path))
        .predict_volume(volume, labels, "", model)
        .mask
    )
    np.testing.assert_array_equal(actual, expected)
    monkeypatch.setattr(runtime, "load_checkpoint", lambda state, artifacts: {"weights": {}})
    inherited = model.model_copy(
        update={
            "read_only": False,
            "inherit_targets": True,
            "config": model.config | {"label_mapping": {"5": 3}},
        }
    )
    np.testing.assert_array_equal(
        runtime.VistaSegmenter({}, Artifacts(tmp_path))
        .predict_volume(volume, labels, "", inherited)
        .mask,
        expected,
    )
    with pytest.raises(DomainError, match="does not support"):
        runtime.VistaSegmenter({}, Artifacts(tmp_path)).predict_volume(
            volume, [Label(id=5, name="Unknown organ", color="#00ff00")], "", model
        )


def test_vista_binary_loss_and_optimizer_continuation(tmp_path, monkeypatch):
    pytest.importorskip("monailabel.monai")
    import torch

    from monailabel.core.models import TrainingMode
    from monailabel.core.ports import TrainingVolume, Volume
    from monailabel.monai import vista_runtime as runtime
    from monailabel.server.storage import Artifacts

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv3d(1, 1, 1)

        def set_auto_grad(self, **kwargs):
            pass

        def forward(self, x, class_vector):
            return self.layer(x).repeat(len(class_vector), 1, 1, 1, 1)

    monkeypatch.setattr(runtime, "vista3d132", Network)
    monkeypatch.setattr(runtime, "load_base", lambda net: None)
    mask = np.zeros((32, 32, 32), np.uint8)
    mask[3:9, 3:9, 3:9] = 5
    mask[19:26, 19:26, 19:26] = 9
    volume = Volume(mask.astype(np.float32)[..., None], np.diag([1.5, 1.5, 1.5, 1]).tolist())
    artifacts = Artifacts(tmp_path)
    calls = []
    forward = Network.forward

    def counted_forward(self, x, class_vector):
        calls.append(x.shape[0])
        return forward(self, x, class_vector)

    monkeypatch.setattr(Network, "forward", counted_forward)
    trainer = runtime.VistaTrainer(
        {
            "device": "cpu",
            "patch_size": 32,
            "epochs": 1,
            "steps_per_epoch": 1,
            "label_mapping": {"5": 3, "9": 1},
        },
        artifacts,
    )
    first = trainer.train_volumes(
        [TrainingVolume(volume, mask)],
        [0, 5, 9],
        TrainingMode.FINE_TUNE,
        {"format": "vista3d-base-v1"},
        lambda _: None,
    )
    # Identical full-size patches must average to the same update, not scale it.
    batch_trainer = runtime.VistaTrainer(
        trainer.config.model_dump(mode="json") | {"batch_size": 3}, artifacts
    )
    batched = batch_trainer.train_volumes(
        [TrainingVolume(volume, mask)],
        [0, 5, 9],
        TrainingMode.FINE_TUNE,
        {"format": "vista3d-base-v1"},
        lambda _: None,
    )
    assert calls == [1] * 4
    assert batched["steps"] == 1 and batched["config"]["batch_size"] == 3
    one, three = (
        runtime.load_checkpoint(first, artifacts),
        runtime.load_checkpoint(batched, artifacts),
    )
    for name in one["weights"]:
        torch.testing.assert_close(one["weights"][name], three["weights"][name])
    trainer = runtime.VistaTrainer(
        trainer.config.model_dump(mode="json")
        | {"batch_size": 2, "weight_decay": 0.02, "learning_rate": 0.001},
        artifacts,
    )
    second = trainer.train_volumes(
        [TrainingVolume(volume, mask)], [0, 5, 9], TrainingMode.CONTINUE, first, lambda _: None
    )
    assert np.isfinite(first["final_loss"]) and np.isfinite(second["final_loss"])
    assert first["steps"] == 1 and second["steps"] == 2
    assert first["checkpoint_key"] != second["checkpoint_key"]
    saved = runtime.load_checkpoint(second, artifacts)
    assert calls == [1] * 6
    assert saved["optimizer"]["param_groups"][0]["weight_decay"] == 0.02
    assert saved["optimizer"]["param_groups"][0]["lr"] == 0.001
    assert all(value["step"] == 2 for value in saved["optimizer"]["state"].values())
    adapted = runtime.VistaTrainer(
        trainer.config.model_dump(mode="json")
        | {
            "label_mapping": {"9": 1},
        },
        artifacts,
    )
    liver_mask = np.where(mask == 9, 9, 0).astype(np.uint8)
    third = adapted.train_volumes(
        [TrainingVolume(volume, liver_mask)],
        [0, 9],
        TrainingMode.FINE_TUNE,
        second,
        lambda _: None,
    )
    assert third["config"]["label_mapping"] == {"9": 1}
    with pytest.raises(DomainError, match="mapping and spacing must match"):
        adapted.train_volumes(
            [TrainingVolume(volume, liver_mask)],
            [0, 9],
            TrainingMode.CONTINUE,
            second,
            lambda _: None,
        )
