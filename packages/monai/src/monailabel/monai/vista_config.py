"""Settings for automatic VISTA3D CT inference and separate project fine-tuning."""

from typing import Literal, Self

from pydantic import Field, model_validator

from monailabel.core.models import Contract


class VistaConfig(Contract):
    device: Literal["cuda", "cpu"] = "cuda"
    patch_size: int = Field(default=128, ge=32, le=192, multiple_of=16)
    spacing: float = Field(default=1.5, ge=0.5, le=4)
    epochs: int = Field(default=5, ge=1, le=100)
    steps_per_epoch: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=1, ge=1, le=32)
    learning_rate: float = Field(default=0.00005, gt=0, le=0.01)
    weight_decay: float = Field(default=0.00001, ge=0, le=1)
    seed: int = Field(default=0, ge=0)
    label_mapping: dict[int, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_mapping(self) -> Self:
        if any(
            not 1 <= key <= 255 or not 1 <= value <= 254
            for key, value in self.label_mapping.items()
        ):
            raise ValueError("VISTA3D mapping requires foreground project and model IDs.")
        if len(set(self.label_mapping.values())) != len(self.label_mapping):
            raise ValueError("VISTA3D target IDs must be unique.")
        return self
