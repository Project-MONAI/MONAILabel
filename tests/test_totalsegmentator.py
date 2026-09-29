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

"""CT/MRI vocabulary, geometry, derived model lineage and training isolation."""

import os

import nibabel as nib
import numpy as np
import pytest
from test_vista3d import reviewed_volumes

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, Snapshot, TrainingMode
from monailabel.core.ports import TrainingVolume, Volume
from monailabel.totalsegmentator.catalog import MODELS, mapping, targets
from monailabel.totalsegmentator.runtime import TotalSegmenter, TotalTrainer


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -2, 256, 0.5, -1])
def test_training_rejects_invalid_or_unreviewed_masks(tmp_path, monkeypatch, value):
    import monailabel.totalsegmentator.runtime as runtime

    monkeypatch.setattr(runtime, "pretrained_weights", lambda *args: tmp_path)
    sample = TrainingVolume(
        Volume(np.zeros((2, 3, 4, 1), np.float32), np.eye(4).tolist()),
        np.full((2, 3, 4), value, dtype=np.float32),
    )
    trainer = TotalTrainer("totalsegmentator-ct", {"label_mapping": {42: 5}}, None)
    with pytest.raises(DomainError, match="project IDs|reviewed voxels"):
        trainer.train_volumes(
            [sample],
            [0, 42],
            TrainingMode.FINE_TUNE,
            {"format": "totalsegmentator-ct-base-v1"},
            lambda _: None,
        )


@pytest.mark.parametrize(
    "provider,count", [("totalsegmentator-ct", 117), ("totalsegmentator-mr", 50)]
)
def test_presets_vocabulary_and_geometry_gating(client, http, provider, count):
    service = http.app.state.services
    service.presets.enabled = True
    service.presets.hosted = {}
    project = client.post("/api/projects", {"name": "Total models"})
    prefix = f"/api/projects/{project['id']}"
    models = client.get(prefix + "/models")
    model = next(m for m in models if m["provider"] == provider)
    assert model["name"] == MODELS[provider][0] and model["read_only"]
    assert len(targets(provider)) == count
    service.presets.ensure(project["id"])
    assert client.get(prefix + "/models") == models
    assert (
        next(m for m in models if m["id"] == project["annotation_model_id"])["provider"]
        == "vista3d"
    )
    record = ModelRecord.model_validate(model)
    assert service.models.configured(record)
    assert service.models.requires_3d(record)
    service.models.validate_targets(record, ["Liver", "kidney left"])
    with pytest.raises(DomainError, match="does not support"):
        service.models.validate_targets(record, ["tumor"])
    assert (
        mapping(provider, [Label(id=8, name="kidney left")])[8] == targets(provider)["kidney_left"]
    )
    if provider.endswith("mr"):
        with pytest.raises(DomainError, match="does not support"):
            service.models.validate_targets(record, ["rib_left_1"])


@pytest.mark.parametrize("provider", list(MODELS))
def test_runtime_remaps_native_ids_and_rejects_changed_grid(tmp_path, monkeypatch, provider):
    import monailabel.totalsegmentator.runtime as runtime

    image = np.ones((7, 8, 9, 1), dtype=np.float32)
    affine = np.diag([-1.7, 2.1, 3.8, 1.0])
    affine[:3, 3] = [32, -12, 55]
    labels = [Label(id=0, name="Background"), Label(id=42, name="Liver")]
    model = ModelRecord(name=provider, provider=provider, label_ids=[0, 42], read_only=True)
    monkeypatch.setattr(runtime, "pretrained_weights", lambda _: tmp_path)
    changed = False

    def worker(directory, request, progress):
        source = nib.load(directory / "image.nii.gz")
        np.testing.assert_allclose(source.affine, affine)
        mask = np.full(image.shape[:-1], targets(provider)["liver"], np.uint8)
        # A non-selected structure must not become a project label by numeric coincidence.
        mask[0] = targets(provider)["spleen"]
        nib.save(nib.Nifti1Image(mask, np.eye(4) if changed else affine), directory / "mask.nii.gz")

    monkeypatch.setattr(runtime, "run_worker", worker)
    result = TotalSegmenter(provider, {}, None).predict_volume(
        Volume(image, affine.tolist()), labels, "", model
    )
    assert np.all(result.mask[0] == 0) and np.all(result.mask[1:] == 42)
    changed = True
    with pytest.raises(DomainError, match="original image grid"):
        TotalSegmenter(provider, {}, None).predict_volume(
            Volume(image, affine.tolist()), labels, "", model
        )


@pytest.mark.parametrize("provider", list(MODELS))
def test_chat_finetuning_keeps_base_and_uses_only_reviewed_training_cases(
    client, http, provider, monkeypatch
):
    service = http.app.state.services
    service.presets.enabled, service.presets.hosted = True, {}
    project = client.post(
        "/api/projects",
        {
            "name": "Total learning",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 5, "name": "Spleen"},
                {"id": 9, "name": "Liver"},
                {"id": 12, "name": "Other"},
            ],
        },
    )
    assets, _ = reviewed_volumes(client, http, project, shared=True)
    prefix = f"/api/projects/{project['id']}"
    base = next(m for m in client.get(prefix + "/models") if m["provider"] == provider)
    calls = []

    class Trainer:
        def train_volumes(self, samples, label_ids, mode, parent_state, progress):
            assert label_ids == [0, 5, 9] and mode == "fine_tune"
            assert parent_state == {"format": provider + "-base-v1"}
            calls.extend(samples)
            assert len(samples) == 2  # The third case has no accepted review.
            assert samples[0].volume.affine == assets[0]["affine"]
            progress(1)
            return {"format": "fixture-checkpoint"}

    monkeypatch.setattr(service.learning.recipes, "trainer", lambda recipe, config: Trainer())
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="fine-tune",
                    name="create_learner",
                    arguments={
                        "recipe": provider,
                        "name": "Organ specialist",
                        "targets": ["Spleen", "Liver"],
                        "start_now": True,
                    },
                )
            ],
        )
    ]
    reply = client.post(
        prefix + "/assistant",
        {"message": f"Fine-tune {MODELS[provider][0]} for spleen and liver now."},
    )
    result = client.wait(reply["job_id"])
    child = service.store.get(ModelRecord, result["model_id"])
    assert child.parent_id == base["id"] and not child.read_only and child.inherit_targets
    assert child.provider == provider and set(child.training_groups) == {"patient-a", "patient-b"}
    assert child.config["label_mapping"] == {
        "5": targets(provider)["spleen"],
        "9": targets(provider)["liver"],
    }
    snapshot = service.store.get(Snapshot, result["snapshot_id"])
    assert {s.group_id for s in snapshot.samples} == {"patient-a", "patient-b"}
    assert service.models.get(base["project_id"], base["id"]).model_dump(mode="json") == base
    assert len(calls) == 2


def test_loss_ignores_unreviewed_voxels_and_preserves_unrequested_anatomy():
    import torch

    from monailabel.totalsegmentator.worker import project_loss

    logits = torch.zeros((1, 4, 2, 2, 2), requires_grad=True)
    target = torch.ones((1, 2, 2, 2), dtype=torch.long)
    target[:, 0] = -1
    loss = project_loss(logits, target, [1])
    loss.backward()
    assert torch.isfinite(loss) and torch.all(logits.grad[:, :, 0] == 0)
    assert torch.any(logits.grad[:, :, 1] != 0)
    # Background includes all non-selected organs, not just upstream class zero.
    alternative = logits.detach().clone()
    alternative[:, 0], alternative[:, 2] = -10, 10
    background = torch.zeros_like(target)
    assert project_loss(alternative, background, [1]) < project_loss(logits, background, [1])


def test_catalog_does_not_import_frameworks_or_download_weights():
    # The catalog does not trigger checkpoint downloads or upstream imports.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from monailabel.totalsegmentator.catalog import targets; "
            "assert len(targets('totalsegmentator-mr')) == 50; "
            "assert 'torch' not in sys.modules; assert 'totalsegmentator' not in sys.modules",
        ],
        check=True,
    )
    assert result.returncode == 0


@pytest.mark.skipif(
    os.environ.get("MONAILABEL_TEST_TOTALSEG_GPU") != "1",
    reason="Opt-in TotalSegmentator GPU integration",
)
@pytest.mark.parametrize(
    "provider,target", [("totalsegmentator-ct", "spleen"), ("totalsegmentator-mr", "prostate")]
)
def test_gpu_finetuning_continuation_and_source_geometry(tmp_path, provider, target):
    import io

    import torch

    from monailabel.server.storage import Artifacts
    from monailabel.totalsegmentator.weights import pretrained_weights

    assert torch.cuda.is_available()
    base = pretrained_weights(provider)
    artifacts = Artifacts(tmp_path / "artifacts")
    labels = [Label(id=0, name="Background"), Label(id=42, name=target)]
    config = {
        "epochs": 1,
        "steps_per_epoch": 1,
        "label_mapping": {str(key): value for key, value in mapping(provider, labels).items()},
    }
    mask = np.zeros((32, 36, 40), np.int16)
    mask[8:24, 9:27, 10:30] = 42
    image = (np.where(mask, 100, 10) + np.random.default_rng(1).normal(0, 2, mask.shape)).astype(
        np.float32
    )
    affine = [[-3, 0, 0, 96], [0, 3, 0, -24], [0, 0, 3, 12], [0, 0, 0, 1]]
    sample = TrainingVolume(Volume(image[..., None], affine), mask)
    trainer = TotalTrainer(provider, config, artifacts)
    state = trainer.train_volumes(
        [sample],
        [0, 42],
        TrainingMode.FINE_TUNE,
        {"format": f"{provider}-base-v1"},
        lambda _: None,
    )
    first_bytes = artifacts.read(state["checkpoint_key"])
    first = torch.load(io.BytesIO(first_bytes), map_location="cpu", weights_only=True)
    assert first["steps"] == 1 and np.isfinite(first["losses"]).all()
    assert first["optimizer"]["state"]
    continued = trainer.train_volumes(
        [sample], [0, 42], TrainingMode.CONTINUE, state, lambda _: None
    )
    second = torch.load(
        io.BytesIO(artifacts.read(continued["checkpoint_key"])),
        map_location="cpu",
        weights_only=True,
    )
    assert second["steps"] == 2
    assert any(
        not torch.equal(first["weights"][key], value) for key, value in second["weights"].items()
    )
    assert artifacts.read(state["checkpoint_key"]) == first_bytes
    model = ModelRecord(
        name="Synthetic anatomy", provider=provider, label_ids=[0, 42], config=config
    )
    for checkpoint in ({}, continued):
        result = TotalSegmenter(provider, checkpoint, artifacts).predict_volume(
            sample.volume, labels, "", model
        )
        assert result.mask.shape == mask.shape
        assert set(np.unique(result.mask)) <= {0, 42}
    # Revalidates every pinned source hash after training and inference.
    assert pretrained_weights(provider) == base
