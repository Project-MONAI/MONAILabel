"""Small execution ports; implementations own their framework and vendor details."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from monailabel.core.models import (
    ClassificationObject,
    Label,
    ModelRecord,
    ObjectClassification,
    SliceScope,
    SpatialPrompt,
    TrainingMode,
)

Image = NDArray[np.float32]
Mask = NDArray[np.uint8]
Progress = Callable[[float], None]


@dataclass(frozen=True)
class TrainingProgress:
    """Optional training messages alongside the existing numeric progress callback."""

    update: Progress
    message: Callable[[str], None]

    def __call__(self, value: float) -> None:
        self.update(value)


def training_message(progress: Progress, message: str) -> None:
    if isinstance(progress, TrainingProgress):
        progress.message(message)


@dataclass(frozen=True)
class Prediction:
    mask: Mask


class Segmenter(Protocol):
    def predict(
        self, image: Image, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction: ...


class PromptedSegmenter(Protocol):
    def predict_prompted(
        self,
        image: Image,
        label_id: int,
        model: ModelRecord,
        spatial: SpatialPrompt,
        plane: SliceScope | None,
        full_volume: bool,
        progress: Progress,
    ) -> Prediction: ...


class Trainer(Protocol):
    def train(
        self,
        samples: Iterable[tuple[Image, Mask]],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]: ...


@dataclass(frozen=True)
class Volume:
    image: Image
    affine: list[list[float]]


@dataclass(frozen=True)
class TrainingVolume:
    volume: Volume
    mask: Mask


@runtime_checkable
class VolumeSegmenter(Protocol):
    def predict_volume(
        self, volume: Volume, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction: ...


@runtime_checkable
class VolumeTrainer(Protocol):
    def train_volumes(
        self,
        samples: Iterable[TrainingVolume],
        label_ids: list[int],
        mode: TrainingMode,
        parent_state: dict[str, JsonValue] | None,
        progress: Progress,
    ) -> dict[str, JsonValue]: ...


class Classifier(Protocol):
    def classify(
        self,
        image: Image,
        objects: list[ClassificationObject],
        categories: list[str],
        prompt: str,
        model: ModelRecord,
    ) -> list[ObjectClassification]: ...


class BinaryArtifacts(Protocol):
    """A provider can persist checkpoints without knowing their storage location."""

    def put(self, content: bytes, suffix: str = "bin") -> str: ...

    def read(self, key: str) -> bytes: ...
