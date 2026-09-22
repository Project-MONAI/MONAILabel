import numpy as np
import pytest

from monailabel.core.models import ModelRecord, TrainingMode
from monailabel.server.storage import Artifacts

pytest.importorskip("monailabel.monai")
pytestmark = pytest.mark.filterwarnings(
    "ignore:The cuda.cudart module is deprecated.*:FutureWarning"
)


def test_unet_excludes_unreviewed_pixels_from_loss_and_keeps_class_255(tmp_path, monkeypatch):
    import torch

    from monailabel.core.ports import IGNORE_LABEL
    from monailabel.monai import runtime

    net = torch.nn.Conv2d(3, 2, 1)
    gradients = []
    net.register_forward_hook(lambda _, __, output: output.register_hook(gradients.append) and None)
    monkeypatch.setattr(runtime, "network", lambda config, classes: net)
    image = np.ones((16, 16, 3), np.float32)
    mask = np.full((16, 16), IGNORE_LABEL, np.int16)
    mask[4:12, 4:12] = 0
    mask[6:10, 6:10] = 255
    state = runtime.UNetTrainer(
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "patch_size": 16,
            "epochs": 1,
            "steps_per_epoch": 1,
            "device": "cpu",
        },
        Artifacts(tmp_path),
    ).train([(image, mask)], [0, 255], TrainingMode.SCRATCH, None, lambda _: None)
    assert np.isfinite(state["final_loss"]) and state["label_ids"] == [0, 255]
    assert len(gradients) == 1
    gradient = gradients[0].numpy()[0]
    assert np.count_nonzero(gradient[:, mask == IGNORE_LABEL]) == 0
    assert np.count_nonzero(gradient[:, mask == 255]) > 0


def test_unet_trains_multiclass_checkpoints_and_preserves_source_shape(tmp_path):
    from monailabel.monai.runtime import UNetSegmenter, UNetTrainer, checkpoint

    artifacts = Artifacts(tmp_path)
    shape = (17, 18, 19)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:8, 2:8, 2:8] = 3
    mask[9:15, 9:15, 9:15] = 7
    image = mask.astype(np.float32)[..., None]
    config = {
        "patch_size": 16,
        "channels": [4, 8, 16, 32],
        "epochs": 1,
        "steps_per_epoch": 2,
        "device": "cpu",
    }
    progress = []
    trainer = UNetTrainer(config, artifacts)
    state = trainer.train([(image, mask)], [0, 3, 7], TrainingMode.SCRATCH, None, progress.append)
    assert state["steps"] == 2
    assert np.isfinite(state["final_loss"])
    assert progress == sorted(progress)
    model = ModelRecord(name="Test U-Net", provider="monai-unet", label_ids=[0, 3, 7])
    original = image.copy()
    first = UNetSegmenter(state, artifacts).predict(image, [], "", model).mask
    second = UNetSegmenter(state, artifacts).predict(image, [], "", model).mask
    assert first.shape == shape
    assert set(np.unique(first)) <= {0, 3, 7}
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, original)
    trainer = UNetTrainer(config | {"learning_rate": 0.002}, artifacts)
    continued = trainer.train(
        [(image, mask)], [0, 3, 7], TrainingMode.CONTINUE, state, lambda _: None
    )
    assert continued["steps"] == 4
    assert continued["checkpoint_key"] != state["checkpoint_key"]
    a, b = checkpoint(state, artifacts), checkpoint(continued, artifacts)
    assert b["optimizer"]["param_groups"][0]["lr"] == 0.002
    assert a["optimizer"]["param_groups"][0]["lr"] == 0.001
    assert any(
        not np.array_equal(a["weights"][key].numpy(), b["weights"][key].numpy())
        for key in a["weights"]
    )
    tuned = trainer.train([(image, mask)], [0, 3, 7], TrainingMode.FINE_TUNE, state, lambda _: None)
    assert tuned["steps"] == 4
    assert tuned["checkpoint_key"] != continued["checkpoint_key"]


def test_create_unet_prompt_creates_setup_without_training_or_changing_defaults(client):
    project = client.post(
        "/api/projects",
        {
            "name": "Spleen",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Spleen", "color": "#ff0000"},
            ],
        },
    )
    body = {
        "message": "create u-net based segementation model for spleen",
        "project_id": project["id"],
    }
    reply = client.post("/api/assistant", body)
    assert reply["job_id"] is None
    learners = client.get(f"/api/projects/{project['id']}/learners")
    assert len(learners) == 1
    assert learners[0]["id"] == reply["data"]["learner_id"]
    assert learners[0]["recipe"] == "monai-unet"
    assert learners[0]["label_ids"] == [0, 1]
    assert client.post("/api/assistant", body)["data"]["learner_id"] == learners[0]["id"]
    assert client.get(f"/api/projects/{project['id']}/models") == []
    assert client.get(f"/api/projects/{project['id']}/jobs") == []
    assert client.get(f"/api/projects/{project['id']}")["defaults"] == {}


def test_neural_training_requires_reviewed_snapshot_and_records_lineage(client, http):
    from monailabel.server.demo import create_demo

    service = http.app.state.services
    demo = create_demo(service.store, service.artifacts, service.datasets)
    project = client.get(f"/api/projects/{demo['project_id']}")
    learner = client.post(
        f"/api/projects/{project['id']}/learners",
        {
            "name": "Small U-Net",
            "recipe": "monai-unet",
            "label_ids": [label["id"] for label in project["labels"]],
            "config": {
                "patch_size": 16,
                "channels": [4, 8, 16, 32],
                "epochs": 1,
                "steps_per_epoch": 1,
                "device": "cpu",
            },
        },
    )
    route = f"/api/projects/{project['id']}/learners/{learner['id']}/train"
    assert http.post(route, json={}).status_code == 422
    assets = client.get(f"/api/projects/{project['id']}/assets")
    chosen = [next(a for a in assets if a["split"] == split) for split in ("train", "validation")]
    for asset in chosen:
        mask = client.get(f"/api/assets/{asset['id']}/fixture")["mask"]
        annotation = client.post(
            f"/api/assets/{asset['id']}/review",
            {"base_revision": 0, "covered_labels": [0, 1, 2], "mask": mask},
        )
        # Submission alone cannot start the learner.
        assert http.post(route, json={}).status_code == 422
        client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    job = client.post(route, {"config": {"learning_rate": 0.002, "steps_per_epoch": 2}})
    result = client.wait(job["id"])
    model = service.store.get(ModelRecord, result["model_id"])
    assert model.config["learning_rate"] == 0.002
    assert model.config["steps_per_epoch"] == 2
    assert client.get(f"/api/projects/{project['id']}/learners")[0]["config"] == learner["config"]
    assert model.learner_id == learner["id"]
    assert model.training_assets == [chosen[0]["id"]]
    assert chosen[1]["group_id"] not in model.training_groups
    pool = next(a for a in assets if a["split"] == "pool")
    output = client.wait(
        client.post(f"/api/assets/{pool['id']}/annotate", {"model_id": model.id})["id"]
    )
    proposal = client.get(f"/api/proposals/{output['proposal_id']}")
    assert proposal["model_ids"] == [model.id]
    assert client.get(f"/api/projects/{project['id']}")["defaults"] == project["defaults"]


def test_unet_scope_and_training_permissions(http, client, seeded):
    demo, assets = seeded
    pid = demo["project_id"]
    project = client.get(f"/api/projects/{pid}")
    response = http.post(
        "/api/assistant",
        json={
            "project_id": pid,
            "message": "create u-net segmentation model for unknown organ",
        },
    )
    assert response.status_code == 422
    assert client.get(f"/api/projects/{pid}/learners") == []
    user = client.post(
        "/api/auth/users", {"username": "neural-annotator", "password": "test-password-1234"}
    )
    client.request(
        "PUT", f"/api/projects/{pid}/members", {"user_id": user["id"], "roles": ["annotator"]}
    )
    http.post(
        "/api/auth/login", json={"username": "neural-annotator", "password": "test-password-1234"}
    )
    assert (
        http.post(
            "/api/assistant", json={"project_id": pid, "message": "create u-net model"}
        ).status_code
        == 403
    )
    assert (
        http.post(
            f"/api/projects/{pid}/learners",
            json={
                "name": "Unauthorized",
                "recipe": "monai-unet",
                "label_ids": [label["id"] for label in project["labels"]],
            },
        ).status_code
        == 403
    )


def test_pool_group_can_be_reserved_for_validation_but_cannot_enter_training(client, http, seeded):
    demo, assets = seeded
    pool = next(a for a in assets if a["split"] == "pool")
    assert client.post(f"/api/assets/{pool['id']}/assign-validation")["split"] == "validation"
    assert http.post(f"/api/assets/{pool['id']}/assign-train").status_code == 409
    assert http.post(f"/api/assets/{pool['id']}/assign-validation").status_code == 409


def test_unet_physical_preprocessing_restores_geometry_and_continues(tmp_path):
    from monailabel.core.ports import TrainingVolume, Volume
    from monailabel.monai.runtime import UNetSegmenter, UNetTrainer
    from monailabel.server.storage import Artifacts

    shape = (17, 18, 19)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:13, 5:14, 4:12] = 7
    image = (mask.astype(np.float32) * 30 - 100)[..., None]
    affine = [[-2, 0, 0, 40], [0, 3, 0, -9], [0, 0, 4, 10], [0, 0, 0, 1]]
    volume = Volume(image, affine)
    config = {
        "patch_size": 16,
        "channels": [4, 8, 16, 32],
        "epochs": 1,
        "steps_per_epoch": 1,
        "device": "cpu",
        "spacing": 2,
        "intensity_window": [-175, 250],
    }
    artifacts = Artifacts(tmp_path)
    trainer = UNetTrainer(config, artifacts)
    state = trainer.train_volumes(
        [TrainingVolume(volume, mask)], [0, 7], TrainingMode.SCRATCH, None, lambda _: None
    )
    assert state["config"]["spacing"] == 2
    model = ModelRecord(
        name="Physical U-Net", provider="monai-unet", label_ids=[0, 7], config=config
    )
    result = UNetSegmenter(state, artifacts).predict_volume(volume, [], "", model)
    assert result.mask.shape == shape
    assert set(np.unique(result.mask)) <= {0, 7}
    np.testing.assert_array_equal(volume.image, image)
    assert volume.affine == affine
    continued = trainer.train_volumes(
        [TrainingVolume(volume, mask)], [0, 7], TrainingMode.CONTINUE, state, lambda _: None
    )
    assert continued["steps"] == 2
    from monailabel.core.errors import DomainError

    with pytest.raises(DomainError, match="preprocessing"):
        UNetTrainer(config | {"spacing": 3}, artifacts).train_volumes(
            [TrainingVolume(volume, mask)], [0, 7], TrainingMode.CONTINUE, state, lambda _: None
        )


def test_rgb_unet_contract_and_parent_layout_are_enforced(tmp_path):
    from monailabel.core.errors import DomainError
    from monailabel.monai.runtime import ImageUNetSegmenter, ImageUNetTrainer

    artifacts = Artifacts(tmp_path)
    mask = np.zeros((21, 25), dtype=np.uint8)
    mask[3:15, 4:18] = 7
    image = np.stack([mask / 7, mask / 14, mask / 21], axis=-1).astype(np.float32)
    config = dict(
        spatial_dims=2,
        in_channels=3,
        patch_size=16,
        channels=[4, 8, 16, 32],
        epochs=1,
        steps_per_epoch=1,
        device="cpu",
    )
    trainer = ImageUNetTrainer(config, artifacts)
    state = trainer.train([(image, mask)], [0, 7], TrainingMode.SCRATCH, None, lambda _: None)
    model = ModelRecord(name="RGB nuclei", provider="monai-unet", label_ids=[0, 7], config=config)
    result = ImageUNetSegmenter(state, artifacts).predict(image, [], "", model)
    assert result.mask.shape == mask.shape and set(np.unique(result.mask)) <= {0, 7}
    with pytest.raises(DomainError, match="channels"):
        ImageUNetSegmenter(state, artifacts).predict(image[..., :1], [], "", model)
    with pytest.raises(DomainError, match="architecture"):
        ImageUNetTrainer(config | {"in_channels": 1}, artifacts).train(
            [(image[..., :1], mask)], [0, 7], TrainingMode.CONTINUE, state, lambda _: None
        )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_unet_batches_patches_and_continues_with_run_weight_decay(
    tmp_path, monkeypatch, dimensions
):
    from monailabel.monai import runtime

    artifacts = Artifacts(tmp_path)
    shape = (16,) * dimensions
    mask = np.zeros(shape, dtype=np.uint8)
    mask[(slice(3, 10),) * dimensions] = 1
    image = mask.astype(np.float32)[..., None]
    seen = []
    build = runtime.network

    def network(config, labels):
        net = build(config, labels)
        net.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
        return net

    monkeypatch.setattr(runtime, "network", network)
    config = dict(
        spatial_dims=dimensions,
        patch_size=16,
        channels=[4, 8, 16, 32],
        epochs=1,
        steps_per_epoch=2,
        batch_size=3,
        weight_decay=0.001,
        device="cpu",
    )
    state = runtime.ImageUNetTrainer(config, artifacts).train(
        [(image, mask)], [0, 1], TrainingMode.SCRATCH, None, lambda _: None
    )
    assert seen == [(3, 1, *shape)] * 2
    assert state["steps"] == 2 and state["config"]["batch_size"] == 3
    saved = runtime.checkpoint(state, artifacts)
    assert all(s["step"] == 2 for s in saved["optimizer"]["state"].values())
    continued = runtime.ImageUNetTrainer(
        config | {"batch_size": 2, "weight_decay": 0.02}, artifacts
    ).train([(image, mask)], [0, 1], TrainingMode.CONTINUE, state, lambda _: None)
    assert seen[-2:] == [(2, 1, *shape)] * 2
    assert continued["steps"] == 4
    assert (
        runtime.checkpoint(continued, artifacts)["optimizer"]["param_groups"][0]["weight_decay"]
        == 0.02
    )
    assert saved["optimizer"]["param_groups"][0]["weight_decay"] == 0.001
