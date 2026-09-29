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

"""Private worker entry point; upstream environment and runtime changes stay in this process."""

import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from monailabel.totalsegmentator.catalog import MODELS
from monailabel.totalsegmentator.config import TotalConfig


def status(root: Path, value: float) -> None:
    temporary = root / "progress.part"
    temporary.write_text(json.dumps({"progress": value}))
    temporary.replace(root / "progress.json")


def project_loss(logits: torch.Tensor, target: torch.Tensor, selected: list[int]) -> torch.Tensor:
    """Marginalize unrequested anatomy into background; ignore unreviewed voxels."""
    others = [i for i in range(logits.shape[1]) if i not in selected]
    scores = torch.cat(
        [logits[:, others].logsumexp(dim=1, keepdim=True), logits[:, selected]], dim=1
    )
    mapped = torch.zeros_like(target)
    mapped[target < 0] = -1
    for index, original in enumerate(selected, 1):
        mapped[target == original] = index
    valid = mapped >= 0
    if not valid.any():
        raise ValueError("Training patch contains no reviewed voxels.")
    ce = F.cross_entropy(scores, mapped, ignore_index=-1)
    probabilities = scores.softmax(dim=1)
    one_hot = F.one_hot(mapped.clamp_min(0), len(selected) + 1).movedim(-1, 1)
    probabilities = probabilities * valid[:, None]
    one_hot = one_hot * valid[:, None]
    axes = (0, 2, 3, 4)
    dice = (2 * (probabilities * one_hot).sum(axes) + 1e-5) / (
        probabilities.sum(axes) + one_hot.sum(axes) + 1e-5
    )
    return ce + 1 - dice[1:].mean()


def prepared_sample(root: Path, index: int, predictor: Any) -> tuple[np.ndarray, np.ndarray]:
    from totalsegmentator.resampling import change_spacing

    # Match upstream fast inference: canonical RAS, linear 3 mm image resampling,
    # then the checkpoint's nnU-Net preprocessor. Masks always use nearest neighbor.
    image = change_spacing(
        nib.as_closest_canonical(nib.load(root / f"image-{index}.nii.gz")),  # type: ignore[no-untyped-call]
        3.0,
        order=1,
        dtype=np.int32,
        nr_cpus=1,
    )
    mask = change_spacing(
        nib.as_closest_canonical(nib.load(root / f"mask-{index}.nii.gz")),  # type: ignore[no-untyped-call]
        3.0,
        order=0,
        dtype=np.int16,
        nr_cpus=1,
    )
    nib.save(image, root / "prepared-image.nii.gz")
    nib.save(mask, root / "prepared-mask.nii.gz")
    processor = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, seg, _ = processor.run_case(
        [str(root / "prepared-image.nii.gz")],
        str(root / "prepared-mask.nii.gz"),
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )
    return np.asarray(data, dtype=np.float32), np.asarray(seg[0], dtype=np.int16)


def train(root: Path, request: dict[str, Any], folder: Path, device: torch.device) -> None:
    from totalsegmentator.nnunet import nnUNetPredictor

    config = TotalConfig.model_validate(request["config"])
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    predictor = nnUNetPredictor(
        device=device,
        use_mirroring=False,
        perform_everything_on_device=False,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(str(folder), use_folds=(0,))
    network = predictor.network
    parent = None
    if (root / "parent.pt").exists():
        parent = torch.load(root / "parent.pt", map_location="cpu", weights_only=True)
        network.load_state_dict(parent["weights"], strict=True)
    network.to(device).train()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if parent and request["mode"] == "continue":
        optimizer.load_state_dict(parent["optimizer"])
        for group in optimizer.param_groups:
            group["lr"], group["weight_decay"] = config.learning_rate, config.weight_decay
    # Keep at most two preprocessed volumes in memory; temporary arrays are job-local.
    cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
    patch_size = predictor.configuration_manager.patch_size
    selected = list(config.label_mapping.values())
    steps = config.epochs * config.steps_per_epoch
    losses = []
    for step in range(steps):
        status(root, 0.02 + 0.94 * step / steps)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.batch_size):
            index = int(rng.integers(1, request["count"] + 1))
            if index not in cache:
                cache[index] = prepared_sample(root, index, predictor)
                if len(cache) > 2:
                    cache.popitem(last=False)
            cache.move_to_end(index)
            data, seg = cache[index]
            positions = np.flatnonzero(np.isin(seg, selected))
            if not len(positions):
                positions = np.flatnonzero(seg >= 0)
            if not len(positions):
                raise ValueError("No reviewed voxels remain after resampling.")
            center = np.unravel_index(int(rng.choice(positions)), seg.shape)
            starts = [
                max(0, min(c - p // 2, n - p))
                for c, p, n in zip(center, patch_size, seg.shape, strict=True)
            ]
            region = tuple(
                slice(start, start + p) for start, p in zip(starts, patch_size, strict=True)
            )
            x, y = data[(slice(None), *region)], seg[region]
            padding = [(0, max(0, int(p) - n)) for p, n in zip(patch_size, y.shape, strict=True)]
            x = np.pad(x, [(0, 0), *padding], mode="edge")
            y = np.pad(y, padding, constant_values=-1)
            inputs = torch.from_numpy(x.copy())[None].to(device)
            target = torch.from_numpy(y.copy()).long()[None].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = network(inputs)
                loss = project_loss(logits.float(), target, selected)
            if not torch.isfinite(loss):
                raise ValueError("TotalSegmentator training produced a non-finite loss.")
            (loss / config.batch_size).backward()  # type: ignore[no-untyped-call]
            total_loss += float(loss.detach()) / config.batch_size
        torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
        optimizer.step()
        losses.append(total_loss)
    optimizer.zero_grad(set_to_none=True)
    network.cpu()
    state = optimizer.state_dict()
    for values in state["state"].values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.cpu()
    torch.save(
        {
            "weights": network.state_dict(),
            "optimizer": state,
            "losses": losses,
            "steps": steps + (parent["steps"] if parent else 0),
        },
        root / "trained.pt",
    )


def main() -> None:
    root = Path(sys.argv[1])
    request = json.loads((root / "request.json").read_text())
    config = TotalConfig.model_validate(request["config"])
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but PyTorch cannot access a GPU.")
    device = torch.device("cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    home = Path(os.environ["TOTALSEG_HOME_DIR"])
    home.mkdir()
    (home / "config.json").write_text(
        json.dumps(
            {
                "totalseg_id": "monailabel-local",
                "send_usage_stats": False,
                "prediction_counter": 0,
                "statistics_disclaimer_shown": True,
            }
        )
    )
    base = Path(request["base"])
    folders = list(base.glob("*__nnUNetPlans__3d_fullres"))
    if len(folders) != 1:
        raise ValueError("Expected one pinned TotalSegmentator network.")
    if request["operation"] == "train":
        train(root, request, folders[0], device)
    else:
        weights = Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
        weights.mkdir()
        if (root / "parent.pt").exists():
            # Materialize a private upstream-compatible checkpoint for inference.
            output = weights / base.name
            shutil.copytree(base, output)
            checkpoint = output / folders[0].name / "fold_0/checkpoint_final.pth"
            upstream = torch.load(checkpoint, map_location="cpu", weights_only=False)
            parent = torch.load(root / "parent.pt", map_location="cpu", weights_only=True)
            upstream["network_weights"] = parent["weights"]
            checkpoint.unlink()
            torch.save(upstream, checkpoint)
        else:
            (weights / base.name).symlink_to(base, target_is_directory=True)
        from totalsegmentator.python_api import totalsegmentator

        totalsegmentator(
            root / "image.nii.gz",
            root / "mask.nii.gz",
            ml=True,
            task=MODELS[request["provider"]][1],
            fast=True,
            device="gpu" if device.type == "cuda" else "cpu",
            nr_thr_resamp=1,
            nr_thr_saving=1,
            quiet=True,
            statistics=False,
        )
    status(root, 0.98)


if __name__ == "__main__":
    main()
