# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""nnU-Net export, app contracts and optional real GPU integration."""

import io
import json
import os

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, TrainingMode
from monailabel.core.ports import TrainingVolume, Volume
from monailabel.monai.nnunet_runtime import (
    DATASET,
    NNUNetSegmenter,
    NNUNetTrainer,
    export_dataset,
)
from monailabel.server.storage import Artifacts


def samples():
    affine = np.array([[0, -1.8, 0, 45], [1.2, 0, 0, -30], [0, 0, 2.3, 12], [0, 0, 0, 1]])
    shape = (24, 28, 32)
    mask = np.zeros(shape, dtype=np.int16)
    mask[7:17, 8:20, 9:23] = 42
    mask[:4] = -1
    rng = np.random.default_rng(72)
    image = rng.normal(30, 10, (*shape, 1)).astype(np.float32)
    image[mask == 42] += 100
    # Unreviewed bright voxels must not affect foreground normalization stats.
    image[mask == -1] = 10000
    return [TrainingVolume(Volume(image, affine.tolist()), mask)]


def test_export_preserves_grid_sparse_ids_and_ignored_coverage(tmp_path):
    example = samples()[0]
    export_dataset(tmp_path, [example], [0, 42], "CT")
    folder = tmp_path / "raw" / DATASET
    image = nib.load(folder / "imagesTr/case_00001_0000.nii.gz")
    mask = nib.load(folder / "labelsTr/case_00001.nii.gz")
    np.testing.assert_allclose(image.affine, example.volume.affine)
    np.testing.assert_array_equal(np.asarray(image.dataobj), example.volume.image[..., 0])
    values = np.asarray(mask.dataobj)
    assert set(np.unique(values)) == {0, 1, 2}
    np.testing.assert_array_equal(values == 2, example.mask == -1)
    np.testing.assert_array_equal(values == 1, example.mask == 42)
    metadata = json.loads((folder / "dataset.json").read_text())
    assert metadata["labels"] == {"background": 0, "project_42": 1, "ignore": 2}
    assert metadata["channel_names"] == {"0": "CT"}


def test_export_keeps_ignore_separate_with_all_uint8_classes(tmp_path):
    mask = np.broadcast_to(np.arange(-1, 256, dtype=np.int16), (2, 2, 257)).copy()
    volume = Volume(np.zeros((*mask.shape, 1), dtype=np.float32), np.eye(4).tolist())
    export_dataset(tmp_path, [TrainingVolume(volume, mask)], list(range(256)), "MRI")
    result = np.asarray(nib.load(tmp_path / "raw" / DATASET / "labelsTr/case_00001.nii.gz").dataobj)
    assert result[0, 0, 0] == 256
    np.testing.assert_array_equal(result[0, 0, 1:], np.arange(256))


@pytest.mark.parametrize("invalid", ["rgb", "grid", "fractional", "empty", "missing_target"])
def test_export_rejects_invalid_training_data(tmp_path, invalid):
    example = samples()[0]
    image, mask = example.volume.image, example.mask.copy()
    if invalid == "rgb":
        image = np.repeat(image, 3, axis=-1)
    elif invalid == "grid":
        mask = mask[:-1]
    elif invalid == "fractional":
        mask = mask.astype(np.float32) + 0.1
    elif invalid == "empty":
        mask[:] = -1
    else:
        mask[mask == 42] = 0
    with pytest.raises(DomainError):
        export_dataset(
            tmp_path, [TrainingVolume(Volume(image, example.volume.affine), mask)], [0, 42], "MRI"
        )


@pytest.mark.parametrize("modality", ["CT", "MRI"])
def test_chat_setup_training_excludes_held_out_cases_and_preserves_lineage(
    client, http, monkeypatch, modality
):
    from test_vista3d import reviewed_volumes

    from monailabel.core.chat import ChatMessage, ToolCall
    from monailabel.core.models import TrainingReport

    service = http.app.state.services
    project = client.post(
        "/api/projects",
        {
            "name": "nnU-Net learning",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 5, "name": "Spleen"},
                {"id": 9, "name": "Liver"},
                {"id": 12, "name": "Other"},
            ],
        },
    )
    assets, _ = reviewed_volumes(client, http, project)
    prefix = f"/api/projects/{project['id']}"
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="create",
                    name="create_learner",
                    arguments={
                        "recipe": "nnunet-v2",
                        "modality": modality,
                        "name": "Organ specialist",
                        "targets": ["Liver"],
                    },
                )
            ],
        )
    ]
    reply = client.post(
        prefix + "/assistant",
        {"message": f"Create {modality} nnU-Net v2 model Organ specialist for liver."},
    )
    assert reply["job_id"] is None
    learner = client.get(prefix + "/learners")[0]
    assert learner["recipe"] == "nnunet-v2" and learner["config"]["modality"] == modality
    assert learner["config"]["epochs"] == 20 and learner["config"]["steps_per_epoch"] == 32
    assert client.get(prefix + "/models") == []
    route = prefix + "/learners/" + learner["id"] + "/train"
    calls = []

    class Trainer:
        def train_volumes(self, examples, label_ids, mode, parent_state, progress):
            calls.append((mode, parent_state))
            assert label_ids == [0, 9]
            assert len(examples) == 1  # patient-b held out; patient-c unreviewed.
            np.testing.assert_array_equal(
                examples[0].volume.image, service.artifacts.array(assets[0]["image_key"])
            )
            progress(1)
            return {"format": "fixture", "generation": len(calls)}

    def score(project, model, examples, labels, progress, check):
        from monailabel.core.models import ModelMetrics

        assert [sample.asset_id for sample in examples] == [assets[1]["id"]]
        return ModelMetrics(mean_dice=0.8, per_class={9: 0.8}), {9: 0.7}

    configs = []

    def trainer(recipe, config):
        configs.append(config)
        return Trainer()

    monkeypatch.setattr(service.learning.recipes, "trainer", trainer)
    monkeypatch.setattr(service.scorer, "score", score)
    epochs = 1000 if modality == "CT" else 500
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="train",
                    name="start_training",
                    arguments={
                        "learner_name": learner["name"],
                        "config": {"epochs": epochs, "steps_per_epoch": 250},
                    },
                )
            ],
        )
    ]
    reply = client.post(
        prefix + "/assistant",
        {"message": f"Train Organ specialist for {epochs} epochs, 250 steps each."},
    )
    result = client.wait(reply["job_id"])
    assert configs[0]["epochs"] == epochs and configs[0]["steps_per_epoch"] == 250
    saved = client.get(prefix + "/learners")[0]
    assert saved["config"] == learner["config"]  # Run overrides do not change setup defaults.
    parent = service.store.get(ModelRecord, result["model_id"])
    assert parent.provider == "nnunet-v2" and parent.training_assets == [assets[0]["id"]]
    assert service.models.requires_3d(parent) and service.models.configured(parent)
    report = service.store.get(TrainingReport, result["training_report_id"])
    assert report.validation_assets == [assets[1]["id"]] and report.metrics.mean_dice == 0.8
    second = client.wait(
        client.post(route, {"mode": "continue", "parent_model_id": parent.id})["id"]
    )
    child = service.store.get(ModelRecord, second["model_id"])
    assert child.parent_id == parent.id and child.state_key != parent.state_key
    assert calls == [("scratch", None), ("continue", {"format": "fixture", "generation": 1})]
    assert service.store.get(ModelRecord, parent.id) == parent
    assert client.get(prefix)["annotation_model_id"] == project["annotation_model_id"]
    response = http.post(route, json={"config": {"modality": "MRI" if modality == "CT" else "CT"}})
    assert response.status_code == 422


def test_modality_required_and_catalog_does_not_load_runtime(client, http):
    from monailabel.monai.nnunet_config import NNUNetConfig

    with pytest.raises(ValueError):
        NNUNetConfig()
    project = client.post(
        "/api/projects",
        {
            "name": "Scan",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Target"},
            ],
        },
    )
    response = http.post(
        f"/api/projects/{project['id']}/learners",
        json={
            "name": "Missing modality",
            "recipe": "nnunet-v2",
            "label_ids": [0, 1],
        },
    )
    assert response.status_code == 422 and "modality" in response.json()["detail"]


def test_worker_cancellation_terminates_its_process_group(tmp_path, monkeypatch):
    import subprocess

    from monailabel.core.errors import Cancelled
    from monailabel.monai import nnunet_runtime as runtime

    events = []

    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            return 0

    def cancel(_):
        raise Cancelled()

    def launch(*args, **kwargs):
        assert kwargs["start_new_session"]
        assert kwargs["env"]["nnUNet_raw"] == str(tmp_path / "raw")
        return Process()

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: events.append(("kill", pid)))
    with pytest.raises(Cancelled):
        runtime.run_worker(tmp_path, {"operation": "train"}, cancel)
    assert events == [("kill", 12345), ("wait", 5)]


@pytest.mark.skipif(
    os.environ.get("MONAILABEL_TEST_NNUNET_GPU") != "1", reason="Opt-in nnU-Net GPU integration"
)
@pytest.mark.parametrize("modality", ["CT", "MRI"])
def test_gpu_scratch_continue_finetune_and_source_grid(tmp_path, modality):
    import torch

    examples = samples()
    artifacts = Artifacts(tmp_path / "artifacts")
    config = {"modality": modality, "epochs": 1, "steps_per_epoch": 2}
    trainer = NNUNetTrainer(config, artifacts)
    progress = []
    state = trainer.train_volumes(examples, [0, 42], TrainingMode.SCRATCH, None, progress.append)
    assert progress == sorted(progress)
    assert state["training"]["completed_epochs"] == 1
    fingerprint = json.loads(artifacts.read(state["fingerprint_key"]))
    assert fingerprint["foreground_intensity_properties_per_channel"]["0"]["max"] < 1000
    plans = json.loads(artifacts.read(state["plans_key"]))
    assert plans["configurations"]["3d_fullres"]["architecture"]["network_class_name"].endswith(
        ".ResidualEncoderUNet"
    )
    expected_normalization = "CTNormalization" if modality == "CT" else "ZScoreNormalization"
    assert plans["configurations"]["3d_fullres"]["normalization_schemes"] == [
        expected_normalization
    ]
    parent_bytes = artifacts.read(state["checkpoint_key"])
    continued = trainer.train_volumes(
        examples, [0, 42], TrainingMode.CONTINUE, state, lambda _: None
    )
    assert continued["training"]["completed_epochs"] == 2
    assert continued["checkpoint_key"] != state["checkpoint_key"]
    assert continued["plans_key"] == state["plans_key"]
    assert artifacts.read(state["checkpoint_key"]) == parent_bytes
    before = torch.load(io.BytesIO(parent_bytes), map_location="cpu", weights_only=False)
    after = torch.load(
        io.BytesIO(artifacts.read(continued["checkpoint_key"])),
        map_location="cpu",
        weights_only=False,
    )
    assert before["optimizer_state"]["state"] and after["optimizer_state"]["state"]
    assert any(
        not torch.equal(before["network_weights"][key], value)
        for key, value in after["network_weights"].items()
    )
    tuned = trainer.train_volumes(examples, [0, 42], TrainingMode.FINE_TUNE, state, lambda _: None)
    assert tuned["training"]["completed_epochs"] == 1
    model = ModelRecord(name="nnU-Net test", provider="nnunet-v2", label_ids=[0, 42], config=config)
    result = NNUNetSegmenter(continued, artifacts).predict_volume(
        examples[0].volume, [Label(id=42, name="Tumor")], "", model
    )
    assert result.mask.shape == examples[0].mask.shape
    assert set(np.unique(result.mask)) <= {0, 42}
    for other in (config | {"modality": "MRI" if modality == "CT" else "CT"},):
        with pytest.raises(DomainError, match="same modality"):
            NNUNetTrainer(other, artifacts).train_volumes(
                examples, [0, 42], TrainingMode.CONTINUE, state, lambda _: None
            )
