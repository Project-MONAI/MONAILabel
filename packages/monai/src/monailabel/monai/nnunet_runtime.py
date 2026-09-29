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

"""nnU-Net adapters; only job-local workers import the upstream runtime."""

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
from pydantic import JsonValue

from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, TrainingMode
from monailabel.core.ports import (
    IGNORE_LABEL,
    BinaryArtifacts,
    Prediction,
    Progress,
    TrainingVolume,
    Volume,
    training_message,
)
from monailabel.monai.nnunet_config import NNUNetConfig

DATASET = "Dataset001_MONAILabel"
FILES = {
    "checkpoint_key": "fold_0/checkpoint_final.pth",
    "plans_key": "plans.json",
    "dataset_key": "dataset.json",
    "fingerprint_key": "dataset_fingerprint.json",
}


def write_volume(path: Path, volume: Volume) -> None:
    image, affine = volume.image, np.asarray(volume.affine)
    if (
        image.ndim != 4
        or image.shape[-1] != 1
        or min(image.shape[:-1]) < 2
        or affine.shape != (4, 4)
        or not np.isfinite(image).all()
        or not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:3, :3])) < 1e-12
    ):
        raise DomainError("nnU-Net v2 requires a finite single-channel 3D volume with geometry.")
    nib.save(nib.Nifti1Image(image[..., 0], affine), path)  # type: ignore[no-untyped-call]


def export_dataset(
    root: Path, samples: Iterable[TrainingVolume], label_ids: list[int], modality: str
) -> int:
    if (
        len(label_ids) < 2
        or label_ids[0] != 0
        or len(set(label_ids)) != len(label_ids)
        or any(identifier < 0 or identifier > 255 for identifier in label_ids)
    ):
        raise DomainError("nnU-Net requires background followed by unique project target IDs.")
    folder = root / "raw" / DATASET
    images, masks = folder / "imagesTr", folder / "labelsTr"
    images.mkdir(parents=True)
    masks.mkdir()
    count, present = 0, set()
    for count, sample in enumerate(samples, 1):
        name = f"case_{count:05d}"
        write_volume(images / f"{name}_0000.nii.gz", sample.volume)
        values = sample.mask
        if (
            values.shape != sample.volume.image.shape[:-1]
            or not np.isfinite(values).all()
            or not np.array_equal(values, np.round(values))
            or np.any((values < 0) & (values != IGNORE_LABEL))
            or np.any(values > 255)
        ):
            raise DomainError("nnU-Net training masks must use project IDs on the source grid.")
        if not np.any(values >= 0):
            raise DomainError("nnU-Net training requires reviewed voxels in every case.")
        # The final contiguous ID is reserved for unreviewed coverage. Other
        # project structures become background for this model's fixed targets.
        # ID 256 is needed for ignored coverage when all uint8 project IDs are used.
        mapped = np.zeros(values.shape, dtype=np.uint16)
        mapped[values == IGNORE_LABEL] = len(label_ids)
        for index, identifier in enumerate(label_ids[1:], 1):
            selected = values == identifier
            if selected.any():
                present.add(identifier)
            mapped[selected] = index
        nib.save(
            nib.Nifti1Image(mapped, np.asarray(sample.volume.affine)),  # type: ignore[no-untyped-call]
            masks / f"{name}.nii.gz",
        )
    if not count or present != set(label_ids[1:]):
        raise DomainError("Accept foreground annotations for every nnU-Net target before training.")
    (folder / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": modality},
                "labels": {"background": 0}
                | {
                    f"project_{identifier}": index
                    for index, identifier in enumerate(label_ids)
                    if index
                }
                | {"ignore": len(label_ids)},
                "numTraining": count,
                "file_ending": ".nii.gz",
                "overwrite_image_reader_writer": "NibabelIOWithReorient",
            }
        )
    )
    return count


def restore(root: Path, state: dict[str, JsonValue], artifacts: BinaryArtifacts) -> None:
    if state.get("format") != "nnunet-v2-v1":
        raise DomainError("Expected a MONAI Label nnU-Net v2 checkpoint.")
    if state.get("nnunet_version") != version("nnunetv2"):
        raise DomainError("This checkpoint requires its recorded nnU-Net v2 runtime version.")
    for key, name in FILES.items():
        identifier = state.get(key)
        if not isinstance(identifier, str):
            raise DomainError(f"nnU-Net checkpoint is missing {name}.")
        target = root / "model" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifacts.read(identifier))


def run_worker(root: Path, request: dict[str, JsonValue], progress: Progress) -> None:
    (root / "request.json").write_text(json.dumps(request))
    environment = os.environ | {
        "nnUNet_raw": str(root / "raw"),
        "nnUNet_preprocessed": str(root / "preprocessed"),
        "nnUNet_results": str(root / "results"),
        "nnUNet_compile": "false",
        "nnUNet_n_proc_DA": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "MPLBACKEND": "Agg",
    }
    with (root / "worker.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "monailabel.monai.nnunet_worker", str(root)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        message = ""
        try:
            while True:
                status = root / "progress.json"
                if status.exists():
                    update = json.loads(status.read_text())
                    if update["message"] != message:
                        message = update["message"]
                        training_message(progress, message)
                    progress(float(update["progress"]))
                else:
                    progress(0)
                try:
                    process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                log.seek(0)
                raise DomainError(f"nnU-Net v2 worker failed: {log.read()[-4000:]}")
            progress(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


class NNUNetTrainer:
    def __init__(self, config: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.config, self.artifacts = NNUNetConfig.model_validate(config), artifacts

    def train_volumes(
        self,
        samples: Iterable[TrainingVolume],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]:
        if (mode == TrainingMode.SCRATCH) != (parent_state is None):
            raise DomainError("Scratch starts without weights; continuation requires a checkpoint.")
        if parent_state and (
            parent_state.get("label_ids") != label_ids
            or NNUNetConfig.model_validate(parent_state.get("config")).modality
            != self.config.modality
        ):
            raise DomainError("Continue nnU-Net with the same modality and ordered target mapping.")
        with tempfile.TemporaryDirectory(prefix="monailabel-nnunet-") as temporary:
            root = Path(temporary)
            training_message(progress, "Exporting training volumes; held-out cases stay excluded.")

            # Checking progress between exports also honors cancellation.
            def checked_samples() -> Iterable[TrainingVolume]:
                for sample in samples:
                    progress(0)
                    yield sample

            count = export_dataset(root, checked_samples(), label_ids, self.config.modality)
            if parent_state:
                restore(root, parent_state, self.artifacts)
            run_worker(
                root,
                {
                    "operation": "train",
                    "config": self.config.model_dump(mode="json"),
                    "mode": mode.value,
                },
                progress,
            )
            state: dict[str, JsonValue] = {
                "format": "nnunet-v2-v1",
                "nnunet_version": version("nnunetv2"),
                "monai_version": version("monai"),
                "label_ids": list(label_ids),
                "config": self.config.model_dump(mode="json"),
                "training_cases": count,
            }
            for key, name in FILES.items():
                path = root / "model" / name
                state[key] = self.artifacts.put(
                    path.read_bytes(), "json" if path.suffix == ".json" else "bin"
                )
            state["training"] = json.loads((root / "training.json").read_text())
            return state


class NNUNetSegmenter:
    def __init__(self, state: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.state, self.artifacts = state, artifacts

    def predict_volume(
        self, volume: Volume, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        mapping = self.state.get("label_ids")
        if not isinstance(mapping, list) or mapping != model.label_ids:
            raise DomainError("nnU-Net checkpoint and model targets do not match.")
        with tempfile.TemporaryDirectory(prefix="monailabel-nnunet-") as temporary:
            root = Path(temporary)
            restore(root, self.state, self.artifacts)
            write_volume(root / "image.nii.gz", volume)
            run_worker(root, {"operation": "predict"}, lambda _: None)
            result = cast(Any, nib.load(root / "mask.nii.gz"))
            values = np.asarray(result.dataobj)
            if values.shape != volume.image.shape[:-1] or not np.allclose(
                result.affine, volume.affine, atol=1e-4, rtol=1e-5
            ):
                raise DomainError("nnU-Net could not restore the original image grid.")
            if not np.isin(values, np.arange(len(mapping))).all():
                raise DomainError("nnU-Net returned invalid class values.")
            mask = np.zeros(values.shape, dtype=np.uint8)
            selected = {label.id for label in labels}
            for index, identifier in enumerate(mapping):
                if identifier in selected:
                    mask[values == index] = identifier
            return Prediction(mask)
