"""VISTA3D CT class prompts and automatic-branch fine-tuning in source geometry.

Uses MONAI's network and transforms, following the pinned official bundle's CT
window, RAS orientation, 1.5mm default spacing, and binary per-class loss.
The interactive point branch is retained but is not trained or exposed here.
"""

import io
from collections import OrderedDict
from collections.abc import Hashable, Iterable
from typing import Any, cast

import monai
import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.inferers.utils import sliding_window_inference
from monai.losses.dice import DiceCELoss
from monai.networks.nets.vista3d import vista3d132
from monai.transforms.compose import Compose
from monai.transforms.post.dictionary import Invertd
from pydantic import JsonValue

from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, TrainingMode
from monailabel.core.ports import (
    BinaryArtifacts,
    Prediction,
    Progress,
    TrainingVolume,
    Volume,
    training_message,
)
from monailabel.monai.runtime import execution
from monailabel.monai.vista_config import VistaConfig
from monailabel.monai.vista_weights import pretrained_weights
from monailabel.monai.volumes import prepare_volume
from monailabel.providers.vista3d import CHECKSUM, REVISION, mapping


def device_for(config: VistaConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise DomainError(
            "VISTA3D requires a CUDA-capable PyTorch installation for this configuration."
        )
    return torch.device(config.device)


def prepare(
    volume: Volume, config: VistaConfig, mask: np.ndarray[Any, Any] | None = None
) -> tuple[dict[Hashable, Any], Compose]:
    return prepare_volume(volume, config.spacing, (-963.8247715525971, 1053.678477684517), mask)


def load_checkpoint(state: dict[str, JsonValue], artifacts: BinaryArtifacts) -> dict[str, Any]:
    key = state.get("checkpoint_key")
    if state.get("format") != "vista3d-v1" or not isinstance(key, str):
        raise DomainError("Unsupported VISTA3D project checkpoint.")
    return cast(
        dict[str, Any],
        torch.load(io.BytesIO(artifacts.read(key)), map_location="cpu", weights_only=True),
    )


def load_base(net: torch.nn.Module) -> None:
    weights = torch.load(pretrained_weights(), map_location="cpu", weights_only=True)
    net.load_state_dict(weights.get("model", weights), strict=True)


class VistaSegmenter:
    def __init__(self, state: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.state, self.artifacts = state, artifacts

    def predict_volume(
        self, volume: Volume, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        config = VistaConfig.model_validate(model.config)
        selected = [label.id for label in labels if label.id]
        indices = (
            mapping(labels) if model.read_only or model.inherit_targets else config.label_mapping
        )
        if not selected or not set(selected) <= set(indices):
            raise DomainError("Choose targets supported by this VISTA3D model.")
        with execution(), torch.inference_mode():
            device = device_for(config)
            data, transform = prepare(volume, config)
            net = vista3d132()
            if model.read_only:
                load_base(net)
            else:
                net.load_state_dict(
                    load_checkpoint(self.state, self.artifacts)["weights"], strict=True
                )
            net.to(device).eval()
            classes = torch.tensor([[indices[i]] for i in selected], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    data["image"].as_tensor().unsqueeze(0),
                    roi_size=(config.patch_size,) * 3,
                    sw_batch_size=1,
                    predictor=lambda x: net(x, class_vector=classes, transpose=True),
                    overlap=0.3,
                    mode="gaussian",
                    padding_mode="replicate",
                    sw_device=device,
                    device="cpu",
                )
            logits = cast(torch.Tensor, logits).float()[0]
            scores, winners = logits.max(dim=0)
            ids = torch.tensor(selected, dtype=torch.uint8)
            result = torch.where(scores > 0, ids[winners], 0).unsqueeze(0)
            data["pred"] = MetaTensor(result)
            restored = Invertd(
                "pred", transform, orig_keys="image", nearest_interp=True, to_tensor=True
            )(data)["pred"]
            mask = np.asarray(restored[0].cpu(), dtype=np.uint8)
            if mask.shape != volume.image.shape[:-1]:
                raise DomainError("VISTA3D could not restore the original image grid.")
            return Prediction(mask)


class VistaTrainer:
    def __init__(self, config: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.config, self.artifacts = VistaConfig.model_validate(config), artifacts

    def train_volumes(
        self,
        samples: Iterable[TrainingVolume],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]:
        samples = list(samples)
        config = self.config
        selected = label_ids[1:]
        if not selected or label_ids[0] != 0 or set(selected) != set(config.label_mapping):
            raise DomainError("VISTA3D training needs an explicit mapping for every target.")
        observed: set[int] = set()
        for sample in samples:
            observed.update(int(i) for i in np.unique(sample.mask))
        if not set(selected) <= observed:
            raise DomainError(
                "Reviewed training data must contain every requested organ. "
                "Missing organs cannot be treated as background."
            )
        if mode != TrainingMode.SCRATCH and parent_state is None:
            raise DomainError("VISTA3D fine-tuning requires a base or project checkpoint.")
        if (
            parent_state
            and parent_state.get("format") == "vista3d-base-v1"
            and mode != TrainingMode.FINE_TUNE
        ):
            raise DomainError("Create a new model by fine-tuning the read-only VISTA3D base.")
        with execution(progress):
            device = device_for(config)
            rng = np.random.default_rng(config.seed)
            torch.manual_seed(config.seed)
            net = vista3d132()
            saved = None
            if parent_state:
                if parent_state.get("format") == "vista3d-base-v1":
                    load_base(net)
                else:
                    previous = VistaConfig.model_validate(parent_state["config"])
                    if (
                        mode == TrainingMode.CONTINUE
                        and previous.label_mapping != config.label_mapping
                    ) or previous.spacing != config.spacing:
                        raise DomainError("Parent VISTA3D target mapping and spacing must match.")
                    saved = load_checkpoint(parent_state, self.artifacts)
                    net.load_state_dict(saved["weights"], strict=True)
            net.set_auto_grad(auto_freeze=False, point_freeze=True)
            net.to(device).train()
            optimizer = torch.optim.AdamW(
                (p for p in net.parameters() if p.requires_grad),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            if mode == TrainingMode.CONTINUE and saved:
                optimizer.load_state_dict(saved["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate
                    group["weight_decay"] = config.weight_decay
            criterion = DiceCELoss(sigmoid=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-5)
            classes = torch.tensor([[config.label_mapping[i]] for i in selected], device=device)
            cache: OrderedDict[int, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = (
                OrderedDict()
            )
            size, steps = config.patch_size, config.epochs * config.steps_per_epoch
            losses = []
            for step in range(steps):
                progress(0.02 + 0.94 * step / steps)
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                # MONAI VISTA3D trains on one patch per forward call. Average gradients
                # across this effective batch before making one optimizer update.
                for batch_index in range(config.batch_size):
                    progress(0.02 + 0.94 * (step + batch_index / config.batch_size) / steps)
                    case = int(rng.integers(len(samples)))
                    if case not in cache:
                        data, _ = prepare(samples[case].volume, config, samples[case].mask)
                        cache[case] = (
                            np.asarray(data["image"][0]),
                            np.asarray(data["label"][0], dtype=np.uint8),
                        )
                        if len(cache) > 2:
                            cache.popitem(last=False)
                    cache.move_to_end(case)
                    image, mask = cache[case]
                    positions = np.flatnonzero(mask == int(rng.choice(selected)))
                    center = (
                        tuple(
                            int(v) for v in np.unravel_index(int(rng.choice(positions)), mask.shape)
                        )
                        if len(positions) and rng.random() < 0.7
                        else tuple(int(rng.integers(n)) for n in mask.shape)
                    )
                    starts = [
                        max(0, min(int(c) - size // 2, n - size))
                        for c, n in zip(center, mask.shape, strict=True)
                    ]
                    region = tuple(slice(start, start + size) for start in starts)
                    patch, target = image[region], mask[region]
                    padding = [(0, max(0, size - n)) for n in patch.shape]
                    patch, target = (
                        np.pad(patch, padding, mode="edge"),
                        np.pad(target, padding, mode="edge"),
                    )
                    x = torch.from_numpy(patch.copy()).unsqueeze(0).unsqueeze(0).to(device)
                    y = torch.from_numpy(target.copy()).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        logits = net(x, class_vector=classes)
                        loss = sum(
                            criterion(logits[[index]].float(), (y == label).float())
                            for index, label in enumerate(selected)
                        ) / len(selected)
                    if not torch.isfinite(loss):
                        raise DomainError("VISTA3D training produced a non-finite loss.")
                    (loss / config.batch_size).backward()
                    step_loss += float(loss.detach().cpu()) / config.batch_size
                optimizer.step()
                losses.append(step_loss)
                if (
                    step == 0
                    or (step + 1) % max(1, steps // 200) == 0
                    or (step + 1) % config.steps_per_epoch == 0
                ):
                    training_message(
                        progress,
                        f"Epoch {step // config.steps_per_epoch + 1}/{config.epochs} · "
                        f"step {step % config.steps_per_epoch + 1}/{config.steps_per_epoch} · "
                        f"loss {step_loss:.6f}",
                    )
            progress(0.98)
            training_message(progress, "Saving checkpoint.")
            total_steps = int(saved["steps"]) + steps if saved else steps
            output = io.BytesIO()
            torch.save(
                {
                    "weights": {k: v.cpu() for k, v in net.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "steps": total_steps,
                },
                output,
            )
            return {
                "format": "vista3d-v1",
                "checkpoint_key": self.artifacts.put(output.getvalue()),
                "config": config.model_dump(mode="json"),
                "base_revision": REVISION,
                "base_sha256": CHECKSUM,
                "steps": total_steps,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "monai_version": str(monai.__version__),
                "torch_version": str(torch.__version__),
                "training_branch": "automatic",
                "device": str(device),
            }
