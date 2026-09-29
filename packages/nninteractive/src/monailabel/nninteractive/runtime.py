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

"""Run nnInteractive without changing MONAI/TotalSegmentator's nnU-Net dependencies."""

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord, SliceScope, SpatialPrompt
from monailabel.core.ports import Image, Prediction, Progress
from monailabel.nninteractive.weights import cache_root, weights


def run_worker(root: Path, progress: Progress) -> None:
    environment = os.environ | {
        "UV_CACHE_DIR": str(cache_root() / "runtime"),
        "UV_LINK_MODE": "copy",
        "nnUNet_compile": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "MPLBACKEND": "Agg",
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root() / "torch"),
    }
    environment.pop("PYTHONPATH", None)
    with (root / "worker.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uv",
                "run",
                "--no-project",
                "--no-config",
                "--locked",
                "--python",
                sys.executable,
                "--script",
                str(Path(__file__).with_name("worker.py")),
                str(root),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                status = root / "progress"
                progress(0.15 + 0.8 * (float(status.read_text()) if status.exists() else 0))
                try:
                    process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                log.seek(0)
                raise DomainError(f"nnInteractive worker failed: {log.read()[-4000:]}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


class NNInteractiveSegmenter:
    def predict_prompted(
        self,
        image: Image,
        label_id: int,
        model: ModelRecord,
        spatial: SpatialPrompt,
        plane: SliceScope | None,
        full_volume: bool,
        progress: Progress,
    ) -> Prediction:
        if image.ndim != 4 or image.shape[-1] != 1 or not np.isfinite(image).all():
            raise DomainError("nnInteractive requires a finite, single-channel CT or MRI volume.")
        if model.provider != "nninteractive" or not 1 <= label_id <= 255:
            raise DomainError("Invalid nnInteractive model or target.")
        if plane is None:
            raise DomainError("Select a source slice in Slicer or OHIF for nnInteractive.")
        base = weights(progress)
        with tempfile.TemporaryDirectory(prefix="monailabel-nninteractive-") as temporary:
            root = Path(temporary)
            np.save(root / "image.npy", image, allow_pickle=False)
            (root / "request.json").write_text(
                json.dumps(
                    {
                        "weights": str(base),
                        "spatial": spatial.model_dump(mode="json"),
                    }
                )
            )
            run_worker(root, progress)
            mask = np.load(root / "mask.npy", allow_pickle=False)
            if mask.shape != image.shape[:-1] or not set(np.unique(mask)) <= {0, 1}:
                raise DomainError("nnInteractive returned invalid source geometry or labels.")
            mask = np.asarray(mask * label_id, dtype=np.uint8)
            if not full_volume:
                selected = [slice(None)] * 3
                selected[plane.axis] = slice(plane.index, plane.index + 1)
                result = np.zeros_like(mask)
                result[tuple(selected)] = mask[tuple(selected)]
                mask = result
            progress(1)
            return Prediction(mask)
