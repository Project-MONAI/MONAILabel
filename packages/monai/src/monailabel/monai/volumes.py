"""Shared physical-grid preprocessing for local scalar-volume recipes."""

from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd

from monailabel.core.errors import DomainError
from monailabel.core.ports import Volume


def prepare_volume(
    volume: Volume,
    spacing: float,
    window: tuple[float, float] | None,
    mask: np.ndarray[Any, Any] | None = None,
) -> tuple[dict[Hashable, Any], Compose]:
    image = volume.image
    affine = np.asarray(volume.affine, dtype=np.float64)
    if image.ndim != 4 or image.shape[-1] != 1 or not np.isfinite(image).all():
        raise DomainError("Preprocessing expects a finite scalar 3D volume.")
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:3, :3])) < 1e-10
    ):
        raise DomainError("Preprocessing requires valid source voxel-to-world geometry.")
    data: dict[Hashable, Any] = {
        "image": MetaTensor(
            torch.from_numpy(image[..., 0].copy()).unsqueeze(0), affine=torch.from_numpy(affine)
        )
    }
    keys = ["image"]
    if mask is not None:
        if mask.shape != image.shape[:-1]:
            raise DomainError("Training mask geometry differs from the source image.")
        data["label"] = MetaTensor(
            torch.from_numpy(mask.copy()).unsqueeze(0), affine=torch.from_numpy(affine)
        )
        keys.append("label")
    transform = Compose(
        [
            Orientationd(keys, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
            Spacingd(
                keys,
                pixdim=(spacing,) * 3,
                mode=["bilinear"] + (["nearest"] if mask is not None else []),
            ),
        ]
    )
    if window:
        transform = Compose(
            [
                *transform.transforms,
                ScaleIntensityRanged(
                    "image", a_min=window[0], a_max=window[1], b_min=0, b_max=1, clip=True
                ),
            ]
        )
    return cast(dict[Hashable, Any], transform(data)), transform
