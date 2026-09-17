"""Small, real CPU baselines for validating the learning loop without downloads."""

from collections.abc import Iterable
from typing import cast

import numpy as np
from pydantic import JsonValue

from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, TrainingMode
from monailabel.core.ports import Image, Mask, Prediction, Progress


class ThresholdSegmenter:
    """Fixed intensity thresholds; intentionally simple, explicitly a demo baseline."""

    def predict(
        self, image: Image, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        thresholds = np.asarray(model.config.get("thresholds", [0.40, 0.82]), dtype=float)
        if thresholds.ndim != 1 or len(thresholds) != len(model.label_ids) - 1:
            raise DomainError("Threshold count must equal the number of model labels minus one.")
        if np.any(np.diff(thresholds) <= 0):
            raise DomainError("Thresholds must be strictly increasing.")
        indices = np.digitize(image.mean(axis=-1), thresholds)
        return Prediction(np.asarray(model.label_ids, dtype=np.uint8)[indices])


class GaussianTrainer:
    """Fit class-conditional diagonal Gaussians from reviewed pixel/voxel features.

    Scratch learns sufficient statistics from zero. Continue adds new samples.
    Fine-tune discounts parent statistics by half before fitting new samples.
    Neither training path reads evaluation masks or fixture references.
    """

    def train(
        self,
        samples: Iterable[tuple[Image, Mask]],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]:
        counts = np.zeros(len(label_ids), dtype=np.float64)
        sums = None
        squares = None
        if parent_state is not None:
            counts = np.asarray(parent_state["counts"], dtype=np.float64)
            sums = np.asarray(parent_state["sums"], dtype=np.float64)
            squares = np.asarray(parent_state["squares"], dtype=np.float64)
            if mode == TrainingMode.FINE_TUNE:
                counts *= 0.5
                sums *= 0.5
                squares *= 0.5
        seen = 0
        for image, mask in samples:
            progress(min(0.9, 0.1 + seen * 0.05))
            features = image.reshape(-1, image.shape[-1]).astype(np.float64)
            if sums is None:
                sums = np.zeros((len(label_ids), features.shape[1]), dtype=np.float64)
                squares = np.zeros_like(sums)
            if features.shape[1] != sums.shape[1]:
                raise DomainError(
                    "Training samples and parent model need matching feature channels."
                )
            assert squares is not None
            flat_mask = mask.reshape(-1)
            for index, label in enumerate(label_ids):
                selected = features[flat_mask == label]
                counts[index] += len(selected)
                sums[index] += selected.sum(axis=0)
                squares[index] += np.square(selected).sum(axis=0)
            seen += 1
        if seen == 0 or sums is None or squares is None:
            raise DomainError("Training requires new, fully reviewed training samples.")
        if np.any(counts == 0):
            missing = [label_ids[i] for i in np.flatnonzero(counts == 0)]
            raise DomainError(f"No reviewed training pixels for labels {missing}.")
        progress(0.95)
        return cast(
            dict[str, JsonValue],
            {
                "label_ids": label_ids,
                "counts": counts.tolist(),
                "sums": sums.tolist(),
                "squares": squares.tolist(),
            },
        )


class GaussianSegmenter:
    def __init__(self, state: dict[str, JsonValue]):
        self.state = state

    def predict(
        self, image: Image, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        counts = np.asarray(self.state["counts"], dtype=np.float64)
        sums = np.asarray(self.state["sums"], dtype=np.float64)
        squares = np.asarray(self.state["squares"], dtype=np.float64)
        means = sums / counts[:, None]
        variances = np.maximum(squares / counts[:, None] - means**2, 1e-5)
        if image.shape[-1] != means.shape[-1]:
            raise DomainError("Image channels do not match this trained model.")
        features = image.reshape(-1, image.shape[-1])
        output = np.empty(len(features), dtype=np.uint8)
        # Chunking keeps voxel-by-class temporary allocations bounded for volumes.
        for start in range(0, len(features), 65536):
            block = features[start : start + 65536]
            distances = np.sum(
                (block[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
                + np.log(variances)[None, :, :],
                axis=-1,
            )
            output[start : start + len(block)] = np.asarray(model.label_ids, dtype=np.uint8)[
                distances.argmin(axis=1)
            ]
        return Prediction(output.reshape(image.shape[:-1]))
