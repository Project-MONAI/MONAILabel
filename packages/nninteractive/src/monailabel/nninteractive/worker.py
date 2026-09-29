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

# /// script
# requires-python = ">=3.12"
# dependencies = ["nninteractive==2.6.0", "nnunetv2==2.8.1", "torch==2.11.0"]
# ///

"""Isolated inference entry point. Inputs and outputs stay on the source IJK grid."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def run(root: Path) -> None:
    request = json.loads((root / "request.json").read_text())
    image = np.load(root / "image.npy", allow_pickle=False)
    session = nnInteractiveInferenceSession(
        device=torch.device("cuda:0"),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=4,
        do_autozoom=True,
        enable_undo=False,
    )
    session.initialize_from_trained_model_folder(request["weights"], use_fold=0)
    # Our source grid is IJK, unlike Slicer's KJI arrays. Do not reverse prompt axes.
    session.set_image(np.ascontiguousarray(np.moveaxis(image, -1, 0)))
    result = np.zeros(image.shape[:-1], dtype=np.uint8)
    session.set_target_buffer(result)
    spatial = request["spatial"]
    interactions = ([spatial["box"]] if spatial["box"] else []) + sorted(
        spatial["points"], key=lambda point: not point["positive"]
    )
    for index, item in enumerate(interactions):
        if isinstance(item, list):
            bounds = [[round(a), round(b) + 1] for a, b in zip(*item, strict=True)]
            session.add_bbox_interaction(bounds, include_interaction=True)
        else:
            session.add_point_interaction(
                tuple(round(v) for v in item["coordinates"]),
                include_interaction=item["positive"],
            )
        temporary = root / "progress.part"
        temporary.write_text(str((index + 1) / len(interactions)))
        temporary.replace(root / "progress")
    np.save(root / "mask.npy", result, allow_pickle=False)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
