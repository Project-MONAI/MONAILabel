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

"""Isolated upstream inference/training workers with immutable checkpoint artifacts."""

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterable
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
from monailabel.totalsegmentator.catalog import VERSION, mapping
from monailabel.totalsegmentator.config import TotalConfig
from monailabel.totalsegmentator.weights import manifest, pretrained_weights


def write_volume(path: Path, volume: Volume) -> None:
    image, affine = volume.image, np.asarray(volume.affine)
    if (
        image.ndim != 4
        or image.shape[-1] != 1
        or affine.shape != (4, 4)
        or not np.isfinite(image).all()
        or not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:3, :3])) < 1e-12
    ):
        raise DomainError("TotalSegmentator requires a finite scalar volume with source geometry.")
    nib.save(nib.Nifti1Image(image[..., 0], affine), path)  # type: ignore[no-untyped-call]


def run_worker(directory: Path, request: dict[str, JsonValue], progress: Progress) -> None:
    (directory / "request.json").write_text(json.dumps(request))
    environment = os.environ | {
        "TOTALSEG_HOME_DIR": str(directory / "config"),
        "TOTALSEG_WEIGHTS_PATH": str(directory / "weights"),
        "nnUNet_raw": str(directory),
        "nnUNet_preprocessed": str(directory),
        "nnUNet_results": str(directory / "weights"),
        "nnUNet_compile": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "MPLBACKEND": "Agg",
    }
    with (directory / "worker.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "monailabel.totalsegmentator.worker", str(directory)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = directory / "progress.json"
            while True:
                if status.exists():
                    progress(float(json.loads(status.read_text())["progress"]))
                else:
                    progress(0)
                try:
                    process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            progress(0.98)
            if process.returncode:
                log.seek(0)
                detail = log.read()[-4000:]
                raise DomainError(f"TotalSegmentator worker failed: {detail}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def request_state(
    directory: Path, provider: str, state: dict[str, JsonValue], artifacts: BinaryArtifacts
) -> None:
    if not state:
        return
    if state.get("format") != "totalsegmentator-v1" or state.get("provider") != provider:
        raise DomainError("TotalSegmentator parent checkpoint belongs to another model.")
    if state.get("base_sha256") != manifest(provider)["sha256"]:
        raise DomainError("TotalSegmentator parent uses different base weights.")
    key = state.get("checkpoint_key")
    if not isinstance(key, str):
        raise DomainError("TotalSegmentator checkpoint is missing.")
    (directory / "parent.pt").write_bytes(artifacts.read(key))


class TotalSegmenter:
    def __init__(self, provider: str, state: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.provider, self.state, self.artifacts = provider, state, artifacts

    def predict_volume(
        self, volume: Volume, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        config = TotalConfig.model_validate(model.config)
        selected = mapping(self.provider, labels)
        if not selected:
            raise DomainError("Choose at least one supported TotalSegmentator structure.")
        base = pretrained_weights(self.provider)
        with tempfile.TemporaryDirectory(prefix="monailabel-totalseg-") as temporary:
            root = Path(temporary)
            write_volume(root / "image.nii.gz", volume)
            request_state(root, self.provider, self.state, self.artifacts)
            run_worker(
                root,
                {
                    "operation": "predict",
                    "provider": self.provider,
                    "base": str(base),
                    "config": config.model_dump(mode="json"),
                },
                lambda _: None,
            )
            result = cast(Any, nib.load(root / "mask.nii.gz"))
            values = np.asarray(result.dataobj)
            if values.shape != volume.image.shape[:-1] or not np.allclose(
                result.affine, volume.affine, atol=1e-4, rtol=1e-5
            ):
                raise DomainError("TotalSegmentator could not restore the original image grid.")
            if not np.isfinite(values).all() or not np.array_equal(values, np.round(values)):
                raise DomainError("TotalSegmentator returned invalid class values.")
            mask = np.zeros(values.shape, dtype=np.uint8)
            for identifier, original in selected.items():
                mask[values == original] = identifier
            return Prediction(mask)


class TotalTrainer:
    def __init__(self, provider: str, config: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.provider, self.config, self.artifacts = (
            provider,
            TotalConfig.model_validate(config),
            artifacts,
        )

    def train_volumes(
        self,
        samples: Iterable[TrainingVolume],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]:
        if mode == TrainingMode.SCRATCH or parent_state is None:
            raise DomainError("Fine-tune a TotalSegmentator base or continue a derived checkpoint.")
        base_parent = parent_state.get("format") == f"{self.provider}-base-v1"
        if base_parent and mode != TrainingMode.FINE_TUNE:
            raise DomainError("Fine-tune the read-only TotalSegmentator base into a new model.")
        if not base_parent and mode == TrainingMode.CONTINUE:
            previous = TotalConfig.model_validate(parent_state["config"])
            if previous.label_mapping != self.config.label_mapping:
                raise DomainError("Continued TotalSegmentator training requires the same targets.")
        if (
            len(label_ids) < 2
            or label_ids[0] != 0
            or len(set(label_ids)) != len(label_ids)
            or set(label_ids[1:]) != set(self.config.label_mapping)
        ):
            raise DomainError("TotalSegmentator training requires a mapping for every target.")
        training_message(progress, "Verifying TotalSegmentator base weights.")
        base = pretrained_weights(self.provider, lambda _: progress(0))
        with tempfile.TemporaryDirectory(prefix="monailabel-totalseg-train-") as temporary:
            root = Path(temporary)
            if not base_parent:
                request_state(root, self.provider, parent_state, self.artifacts)
            count = 0
            observed: set[int] = set()
            for count, sample in enumerate(samples, 1):
                progress(0.01)
                write_volume(root / f"image-{count}.nii.gz", sample.volume)
                if sample.mask.shape != sample.volume.image.shape[:-1]:
                    raise DomainError("Training image and reference mask geometry differ.")
                if (
                    not np.isfinite(sample.mask).all()
                    or not np.array_equal(sample.mask, np.round(sample.mask))
                    or np.any((sample.mask < 0) & (sample.mask != IGNORE_LABEL))
                    or np.any(sample.mask > 255)
                ):
                    raise DomainError(
                        "Training masks require project IDs or -1 for ignored voxels."
                    )
                if not np.any(sample.mask >= 0):
                    raise DomainError("Training requires reviewed voxels in every case.")
                observed.update(int(value) for value in np.unique(sample.mask))
                # Unrequested project labels remain background; -1 retains excluded coverage.
                mask = np.where(sample.mask == IGNORE_LABEL, IGNORE_LABEL, 0).astype(np.int16)
                for label, original in self.config.label_mapping.items():
                    mask[sample.mask == label] = original
                nib.save(
                    nib.Nifti1Image(mask, np.asarray(sample.volume.affine)),  # type: ignore[no-untyped-call]
                    root / f"mask-{count}.nii.gz",
                )
            if not count or not set(label_ids[1:]) <= observed:
                raise DomainError("Reviewed training data must contain every requested structure.")
            training_message(progress, "Fine-tuning TotalSegmentator on reviewed volume patches.")
            run_worker(
                root,
                {
                    "operation": "train",
                    "provider": self.provider,
                    "base": str(base),
                    "config": self.config.model_dump(mode="json"),
                    "count": count,
                    "mode": mode.value,
                },
                progress,
            )
            training_message(progress, "Saving the derived TotalSegmentator checkpoint.")
            progress(0.99)
            return {
                "format": "totalsegmentator-v1",
                "provider": self.provider,
                "upstream_version": VERSION,
                "base_sha256": manifest(self.provider)["sha256"],
                "config": self.config.model_dump(mode="json"),
                "checkpoint_key": self.artifacts.put((root / "trained.pt").read_bytes()),
            }
