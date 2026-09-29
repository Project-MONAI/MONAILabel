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

"""Source geometry, isolated prompt replay, cancellation and pinned-weight integrity."""

import hashlib
import json
import runpy
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord, PromptPoint, SliceScope, SpatialPrompt
from monailabel.nninteractive import runtime, weights


def test_worker_replays_hints_in_source_axes(tmp_path, monkeypatch):
    calls = []
    image = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9, 1) - 200
    np.save(tmp_path / "image.npy", image)
    request = {
        "weights": "verified-weights",
        "spatial": {
            "box": [[1, 2, 4], [3, 5, 4]],
            "points": [
                {"coordinates": [4, 6, 7], "positive": False},
                {"coordinates": [2, 3, 4], "positive": True},
            ],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request))

    class Session:
        def __init__(self, **kwargs):
            assert str(kwargs["device"]) == "cuda:0"

        def initialize_from_trained_model_folder(self, path, use_fold):
            assert path == "verified-weights" and use_fold == 0

        def set_image(self, values):
            np.testing.assert_array_equal(values, image[..., 0][None])

        def set_target_buffer(self, result):
            self.result = result

        def add_bbox_interaction(self, bounds, include_interaction):
            calls.append((bounds, include_interaction))

        def add_point_interaction(self, coordinates, include_interaction):
            calls.append((coordinates, include_interaction))
            self.result[coordinates] = include_interaction

    monkeypatch.setitem(
        sys.modules,
        "nnInteractive.inference.inference_session",
        SimpleNamespace(nnInteractiveInferenceSession=Session),
    )
    worker = runpy.run_path(str(Path(runtime.__file__).with_name("worker.py")))
    worker["run"](tmp_path)
    assert calls == [([[1, 4], [2, 6], [4, 5]], True), ((2, 3, 4), True), ((4, 6, 7), False)]
    result = np.load(tmp_path / "mask.npy")
    assert result.shape == (5, 7, 9) and result[2, 3, 4] == 1 and result.sum() == 1
    assert (tmp_path / "progress").read_text() == "1.0"


@pytest.mark.parametrize("full", [True, False])
def test_result_uses_project_label_and_requested_scope(monkeypatch, tmp_path, full):
    image = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9, 1) - 200
    spatial = SpatialPrompt(points=[PromptPoint(coordinates=[2, 3, 4])])
    monkeypatch.setattr(runtime, "weights", lambda progress: tmp_path)

    def run(root, progress):
        np.testing.assert_array_equal(np.load(root / "image.npy"), image)
        assert json.loads((root / "request.json").read_text())["spatial"] == spatial.model_dump()
        np.save(root / "mask.npy", np.ones(image.shape[:-1], np.uint8))

    monkeypatch.setattr(runtime, "run_worker", run)
    result = (
        runtime.NNInteractiveSegmenter()
        .predict_prompted(
            image,
            7,
            ModelRecord(
                project_id="p", name="nnInteractive", provider="nninteractive", label_ids=[0]
            ),
            spatial,
            SliceScope(axis=1, index=3),
            full,
            lambda _: None,
        )
        .mask
    )
    assert result.shape == image.shape[:-1] and result.dtype == np.uint8
    assert np.all(result[:, 3, :] == 7)
    assert np.all(result[:, :3, :] == (7 if full else 0))
    assert np.all(result[:, 4:, :] == (7 if full else 0))


def test_cancel_stops_entire_worker_process_group(tmp_path, monkeypatch):
    killed = []

    class Process:
        pid = 42

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    def cancel(_):
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        runtime.run_worker(tmp_path, cancel)
    assert killed == [(42, signal.SIGTERM)]


def test_corrupt_cached_weights_are_rejected_before_loading(tmp_path, monkeypatch):
    content = b"verified checkpoint fixture"
    root = tmp_path / weights.REVISION
    root.mkdir()
    path = root / "checkpoint.pth"
    path.write_bytes(content)
    monkeypatch.setattr(weights, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        weights, "FILES", {path.name: (len(content), hashlib.sha256(content).hexdigest())}
    )
    assert weights.weights(lambda _: None) == root
    path.write_bytes(b"x" * len(content))
    with pytest.raises(DomainError, match="checksum"):
        weights.weights(lambda _: None)
