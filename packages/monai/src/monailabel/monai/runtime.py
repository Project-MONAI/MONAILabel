"""Bounded 2D/3D patch training and source-grid inference using MONAI's U-Net.

Only accepted training arrays enter this port. Checkpoints include the preprocessing
recipe, weights, optimizer state, and dependency versions. No model hub is executed.
"""

import io
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import monai
import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.inferers.utils import sliding_window_inference
from monai.losses.dice import DiceCELoss
from monai.networks.nets.unet import UNet
from monai.transforms.post.dictionary import Invertd
from pydantic import JsonValue

from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, TrainingMode
from monailabel.core.ports import (
    IGNORE_LABEL,
    BinaryArtifacts,
    Image,
    Prediction,
    Progress,
    TrainingMask,
    TrainingVolume,
    Volume,
    training_message,
)
from monailabel.monai.config import UNetConfig
from monailabel.monai.volumes import prepare_volume

# Serialize this local runtime's GPU use and its changes to PyTorch RNG/thread state.
_RUNTIME_LOCK = threading.Lock()


@contextmanager
def execution(progress: Progress | None = None) -> Iterator[None]:
    while not _RUNTIME_LOCK.acquire(timeout=0.25):
        if progress:
            progress(0)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(threads, 4))
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(threads)
        _RUNTIME_LOCK.release()


def device_for(config: UNetConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise DomainError("CUDA was selected but PyTorch cannot access a GPU.")
    return torch.device("cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu")


def network(config: UNetConfig, classes: int) -> UNet:
    return UNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=classes,
        channels=config.channels,
        strides=(2, 2, 2),
        num_res_units=2,
        norm=("GROUP", {"num_groups": 1}),
    )


def validate_image(image: Image, config: UNetConfig) -> None:
    if image.ndim != config.spatial_dims + 1 or image.shape[-1] != config.in_channels:
        raise DomainError(
            f"This U-Net requires {config.spatial_dims}D images with {config.in_channels} channels."
        )
    if not np.isfinite(image).all():
        raise DomainError("U-Net input contains non-finite intensities.")


def statistics(image: Image) -> tuple[float, float]:
    return float(image.mean(dtype=np.float64)), max(float(image.std(dtype=np.float64)), 1e-6)


def normalize(image: Image, stats: tuple[float, float]) -> Image:
    mean, scale = stats
    return cast(Image, np.clip((np.asarray(image, dtype=np.float32) - mean) / scale, -5, 5))


def checkpoint(state: dict[str, JsonValue], artifacts: BinaryArtifacts) -> dict[str, Any]:
    key = state.get("checkpoint_key")
    if not isinstance(key, str) or state.get("format") != "monai-unet-v1":
        raise DomainError("Unsupported U-Net checkpoint format.")
    # Generated tensor/primitive checkpoints only; arbitrary Python objects are never loaded.
    return cast(
        dict[str, Any],
        torch.load(io.BytesIO(artifacts.read(key)), map_location="cpu", weights_only=True),
    )


class ImageUNetTrainer:
    def __init__(self, config: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.config = UNetConfig.model_validate(config)
        self.artifacts = artifacts

    def train(
        self,
        samples: Iterable[tuple[Image, TrainingMask]],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]:
        with execution(progress):
            return self._train(
                samples if isinstance(samples, Sequence) else list(samples),
                label_ids,
                mode,
                parent_state,
                progress,
            )

    def _train(
        self,
        samples: Sequence[tuple[Image, TrainingMask]],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]:
        if not label_ids or label_ids[0] != 0 or len(set(label_ids)) != len(label_ids):
            raise DomainError("U-Net labels must be unique, with background 0 first.")
        if not samples:
            raise DomainError("U-Net training requires reviewed training volumes.")
        config = self.config
        device = device_for(config)
        rng = np.random.default_rng(config.seed)
        torch.set_rng_state(torch.Generator().manual_seed(config.seed).get_state())
        net = network(config, len(label_ids)).to(device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        prior_steps = 0
        if parent_state is not None:
            previous = UNetConfig.model_validate(parent_state["config"])
            if (
                previous.channels != config.channels
                or previous.spatial_dims != config.spatial_dims
                or previous.in_channels != config.in_channels
                or parent_state["label_ids"] != label_ids
            ):
                raise DomainError("Parent U-Net architecture and ordered labels must match.")
            if mode == TrainingMode.CONTINUE and (
                previous.spacing != config.spacing
                or previous.intensity_window != config.intensity_window
            ):
                raise DomainError(
                    "Continued U-Net training must retain its preprocessing settings."
                )
            saved = checkpoint(parent_state, self.artifacts)
            net.load_state_dict(saved["weights"], strict=True)
            if mode == TrainingMode.CONTINUE:
                optimizer.load_state_dict(saved["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate
                    group["weight_decay"] = config.weight_decay
            prior_steps = int(saved["steps"])
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
        lookup = np.zeros(256, dtype=np.int64)
        for index, label in enumerate(label_ids):
            lookup[label] = index
        locations = []
        scales = []
        coverage_locations = []
        seen: set[int] = set()
        for image, mask in samples:
            progress(0.01)
            validate_image(image, config)
            if image.shape[:-1] != mask.shape:
                raise DomainError("Training mask geometry does not match its image.")
            scales.append(statistics(image))
            valid = np.flatnonzero(mask != IGNORE_LABEL) if np.any(mask == IGNORE_LABEL) else None
            if valid is not None and not len(valid):
                raise DomainError("A training region has no reviewed pixels.")
            coverage_locations.append(
                rng.choice(
                    valid,
                    min(len(valid), max(16, min(10000, 1_000_000 // len(samples)))),
                    replace=False,
                )
                if valid is not None
                else None
            )
            targets = []
            for label in label_ids[1:]:
                positions = np.flatnonzero(mask == label)
                if len(positions):
                    seen.add(label)
                    # Keep only a bounded set of candidate patch centers per class/case.
                    limit = max(16, min(10000, 1_000_000 // len(samples)))
                    positions = rng.choice(positions, min(len(positions), limit), replace=False)
                    targets.append(positions)
            locations.append(targets)
        if set(label_ids[1:]) - seen:
            raise DomainError(
                "Reviewed training data must contain every selected foreground label."
            )
        size = config.patch_size
        steps = config.epochs * config.steps_per_epoch
        losses = []
        net.train()
        for step in range(steps):
            progress(0.02 + 0.94 * step / steps)
            patches, targets = [], []
            for batch_index in range(config.batch_size):
                progress(0.02 + 0.94 * (step + batch_index / config.batch_size) / steps)
                case = int(rng.integers(len(samples)))
                image, mask = samples[case]
                shape = mask.shape
                if locations[case] and rng.random() < 0.5:
                    candidates = locations[case][int(rng.integers(len(locations[case])))]
                    center = tuple(
                        int(c) for c in np.unravel_index(int(rng.choice(candidates)), shape)
                    )
                else:
                    valid = coverage_locations[case]
                    center = (
                        tuple(int(c) for c in np.unravel_index(int(rng.choice(valid)), shape))
                        if valid is not None
                        else tuple(int(rng.integers(n)) for n in shape)
                    )
                starts = [
                    max(0, min(int(c) - size // 2, n - size))
                    for c, n in zip(center, shape, strict=True)
                ]
                region = tuple(slice(start, start + size) for start in starts)
                patch = normalize(image[region], scales[case])
                target = lookup[mask[region]]
                target[mask[region] == IGNORE_LABEL] = IGNORE_LABEL
                padding = [(0, max(0, size - n)) for n in target.shape]
                patch = np.pad(patch, padding + [(0, 0)], mode="edge")
                target = np.pad(target, padding, mode="edge")
                # Image and target undergo identical random axis flips.
                for axis in range(config.spatial_dims):
                    if rng.random() < 0.5:
                        patch, target = np.flip(patch, axis), np.flip(target, axis)
                patches.append(np.moveaxis(patch, -1, 0).copy())
                targets.append(target.copy())
            x = torch.from_numpy(np.stack(patches)).to(device)
            y = torch.from_numpy(np.stack(targets)).unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            if torch.any(y == IGNORE_LABEL):
                # Compute Dice and cross-entropy only on reviewed pixels.
                valid_pixels = y[:, 0] != IGNORE_LABEL
                selected = logits.movedim(1, -1)[valid_pixels].T.unsqueeze(0)
                loss = loss_function(selected, y[:, 0][valid_pixels].reshape(1, 1, -1))
            else:
                loss = loss_function(logits, y)
            if not torch.isfinite(loss):
                raise DomainError("U-Net training diverged; lower the learning rate and retry.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (
                step == 0
                or (step + 1) % max(1, steps // 200) == 0
                or (step + 1) % config.steps_per_epoch == 0
            ):
                training_message(
                    progress,
                    f"Epoch {step // config.steps_per_epoch + 1}/{config.epochs} · "
                    f"step {step % config.steps_per_epoch + 1}/{config.steps_per_epoch} · "
                    f"loss {losses[-1]:.6f}",
                )
        progress(0.98)
        training_message(progress, "Saving checkpoint.")
        output = io.BytesIO()
        torch.save(
            {
                "weights": {k: v.cpu() for k, v in net.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "steps": prior_steps + steps,
            },
            output,
        )
        return {
            "format": "monai-unet-v1",
            "checkpoint_key": self.artifacts.put(output.getvalue()),
            "config": config.model_dump(mode="json"),
            "label_ids": cast(list[JsonValue], list(label_ids)),
            "normalization": "per-image-zscore-clip-5",
            "monai_version": str(monai.__version__),
            "torch_version": str(torch.__version__),
            "device": str(device),
            "steps": prior_steps + steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        }


class ImageUNetSegmenter:
    def __init__(self, state: dict[str, JsonValue], artifacts: BinaryArtifacts):
        self.state, self.artifacts = state, artifacts

    def predict(
        self, image: Image, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        config = UNetConfig.model_validate(self.state["config"])
        validate_image(image, config)
        with execution(), torch.inference_mode():
            device = device_for(config)
            net = network(config, len(model.label_ids)).to(device)
            net.load_state_dict(checkpoint(self.state, self.artifacts)["weights"], strict=True)
            net.eval()
            values = normalize(image, statistics(image))
            x = torch.from_numpy(np.moveaxis(values, -1, 0).copy()).unsqueeze(0)
            logits = sliding_window_inference(
                x,
                roi_size=(config.patch_size,) * config.spatial_dims,
                sw_batch_size=1,
                predictor=net,
                overlap=0.25,
                mode="gaussian",
                sw_device=device,
                device="cpu",
            )
            indices = cast(torch.Tensor, logits).argmax(dim=1)[0].numpy()
            mask = np.asarray(model.label_ids, dtype=np.uint8)[indices]
            return Prediction(mask)


class UNetTrainer(ImageUNetTrainer):
    """Volume port adds optional physical preprocessing to the shared trainer."""

    def train_volumes(
        self,
        samples: Iterable[TrainingVolume],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]:
        with execution(progress):
            prepared = []
            for sample in samples:
                progress(0.01)
                if self.config.spacing is None:
                    prepared.append((sample.volume.image, sample.mask))
                    continue
                data, _ = prepare_volume(
                    sample.volume, self.config.spacing, self.config.intensity_window, sample.mask
                )
                prepared.append(
                    (
                        np.asarray(data["image"][0], dtype=np.float32)[..., None],
                        np.asarray(data["label"][0], dtype=np.uint8),
                    )
                )
            return self._train(prepared, label_ids, mode, parent_state, progress)


class UNetSegmenter(ImageUNetSegmenter):
    """Volume port restores predictions onto the original source geometry."""

    def predict_volume(
        self, volume: Volume, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        config = UNetConfig.model_validate(self.state["config"])
        if config.spacing is None:
            return self.predict(volume.image, labels, prompt, model)
        data, transform = prepare_volume(volume, config.spacing, config.intensity_window)
        image = np.asarray(data["image"][0], dtype=np.float32)[..., None]
        result = self.predict(image, labels, prompt, model)
        data["pred"] = MetaTensor(torch.from_numpy(result.mask.copy()).unsqueeze(0))
        restored = Invertd(
            "pred", transform, orig_keys="image", nearest_interp=True, to_tensor=True
        )(data)["pred"]
        mask = np.asarray(restored[0].cpu(), dtype=np.uint8)
        if mask.shape != volume.image.shape[:-1]:
            raise DomainError("U-Net could not restore the source volume grid.")
        return Prediction(mask)
