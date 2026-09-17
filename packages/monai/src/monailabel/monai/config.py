"""Versioned recipe settings, independent of the server and PyTorch imports."""

from typing import Literal, Self

from pydantic import Field, model_validator

from monailabel.core.models import Contract


class UNetConfig(Contract):
    spatial_dims: Literal[2, 3] = 3
    in_channels: Literal[1, 3] = 1
    patch_size: int = Field(default=64, ge=16, le=128, multiple_of=8)
    channels: list[int] = Field(
        default_factory=lambda: [16, 32, 64, 128], min_length=4, max_length=4
    )
    epochs: int = Field(default=20, ge=1, le=1000)
    steps_per_epoch: int = Field(default=32, ge=1, le=1000)
    batch_size: int = Field(default=1, ge=1, le=32)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    weight_decay: float = Field(default=0, ge=0, le=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    spacing: float | None = Field(default=None, ge=0.5, le=4)
    intensity_window: tuple[float, float] | None = None

    @model_validator(mode="after")
    def channel_sizes(self) -> Self:
        if self.spatial_dims == 3 and self.in_channels != 1:
            raise ValueError("3D training requires scalar volumes.")
        if self.spatial_dims == 2 and self.spacing is not None:
            raise ValueError("Physical volume spacing is not a 2D image setting.")
        if any(c < 4 or c > 256 for c in self.channels):
            raise ValueError("Each channel width must be between 4 and 256.")
        if self.intensity_window and (
            self.spacing is None or self.intensity_window[0] >= self.intensity_window[1]
        ):
            raise ValueError("An intensity window requires physical spacing and increasing bounds.")
        return self
