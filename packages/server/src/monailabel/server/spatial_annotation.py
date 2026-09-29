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

"""Validate viewer hints and merge promptable-model output into an immutable proposal."""

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.geometry import region_pixels
from monailabel.core.models import AnnotateRequest, Asset, ModelRecord, SpatialPrompt
from monailabel.core.ports import Image, Mask, Progress, PromptedSegmenter
from monailabel.providers.spatial import MODELS


def validate(
    asset: Asset, request: AnnotateRequest, model: ModelRecord, labels: list[int]
) -> SpatialPrompt:
    spec = MODELS[model.provider].interaction
    if len(labels) != 1:
        raise DomainError(
            "This model annotates one object at a time. Name one target for the box or points."
        )
    if request.image_tiling:
        raise DomainError(
            "This model needs an object box or points. Select a region; "
            "whole-slide automatic nuclei detection needs a nuclei model."
        )
    if spec.volume_only and asset.kind != "volume3d":
        raise DomainError(f"{model.name} requires a medical volume; use SAM 2.1 for 2D images.")
    if request.all_slices and "full" not in spec.output_scopes:
        raise DomainError(f"{model.name} accepts only a selected slice.")
    if asset.kind == "volume3d" and (
        request.slice is None or (spec.intensity_window and request.slice.window is None)
    ):
        raise DomainError(
            "Open a source slice in Slicer or OHIF to supply the seed plane and intensity window."
        )
    spatial = request.spatial_prompt
    crop = request.image_region
    if spatial is None and crop:
        spatial = SpatialPrompt(
            box=[[crop.y, crop.x], [crop.y + crop.height - 1, crop.x + crop.width - 1]]
        )
    if spatial is None:
        raise DomainError(
            "This model needs a spatial hint. Select an object box or add positive/negative "
            "points in the viewer, then retry."
        )
    if spatial.box and "box" not in spec.inputs:
        raise DomainError(f"{model.name} does not accept boxes.")
    for point in spatial.points:
        kind = "positive_point" if point.positive else "negative_point"
        if kind not in spec.inputs:
            raise DomainError(f"{model.name} does not accept {kind.replace('_', ' ')} inputs.")
    coordinates = [point.coordinates for point in spatial.points] + (spatial.box or [])
    if any(
        len(point) != len(asset.spatial_shape)
        or any(v > size - 1 for v, size in zip(point, asset.spatial_shape, strict=True))
        for point in coordinates
    ):
        raise DomainError("Move or clear hints outside the source image before running inference.")
    plane = request.slice
    volume_prompts = spec.prompt_scope == "volume" and request.all_slices
    axes = [axis for axis in range(len(asset.spatial_shape)) if plane is None or axis != plane.axis]
    if volume_prompts and spatial.box:
        axes = [
            axis
            for axis in range(len(asset.spatial_shape))
            if round(spatial.box[0][axis]) != round(spatial.box[1][axis])
        ]
        if len(axes) < 2:
            raise DomainError("Draw a box with nonzero width and height.")
    if spatial.box and any(spatial.box[0][axis] >= spatial.box[1][axis] for axis in axes):
        raise DomainError("Draw a box with nonzero width and height.")
    if (
        spec.prompt_scope == "volume"
        and spec.inputs.get("box") == "slice"
        and spatial.box
        and all(round(a) != round(b) for a, b in zip(*spatial.box, strict=True))
    ):
        raise DomainError(f"{model.name} needs a box on one slice, not a 3D ROI.")
    if plane:
        if any(
            abs(p.coordinates[plane.axis] - plane.index) > 0.5
            and (
                not volume_prompts
                or spec.inputs["positive_point" if p.positive else "negative_point"] == "slice"
            )
            for p in spatial.points
        ):
            raise DomainError("Place points on the selected source slice.")
        if (
            spatial.box
            and not volume_prompts
            and not spatial.box[0][plane.axis] - 0.5
            <= plane.index
            <= spatial.box[1][plane.axis] + 0.5
        ):
            raise DomainError("Select a slice intersecting the box or ROI.")
    return spatial


def predict(
    provider: PromptedSegmenter,
    image: Image,
    mask: Mask,
    model: ModelRecord,
    request: AnnotateRequest,
    spatial: SpatialPrompt,
    label: int,
    progress: Progress,
) -> Mask:
    crop, plane = request.image_region, request.slice
    full_volume = request.all_slices
    region: list[slice | int] = [slice(None)] * mask.ndim
    input_image = image
    hints = spatial
    if crop:
        region = [slice(crop.y, crop.y + crop.height), slice(crop.x, crop.x + crop.width)]
        input_image = image[tuple(region)]
        if crop.runs is not None:
            input_image = np.where(region_pixels(crop)[..., None], input_image, 0)
        hints = spatial.model_copy(
            update={
                "points": [
                    p.model_copy(
                        update={
                            "coordinates": [p.coordinates[0] - crop.y, p.coordinates[1] - crop.x]
                        }
                    )
                    for p in spatial.points
                ],
                "box": [[p[0] - crop.y, p[1] - crop.x] for p in spatial.box]
                if spatial.box
                else None,
            }
        )
        if any(
            v < 0 or v > size - 1
            for p in [x.coordinates for x in hints.points] + (hints.box or [])
            for v, size in zip(p, input_image.shape[:2], strict=True)
        ):
            raise DomainError("Spatial hints must lie inside the selected region.")
    elif plane and not full_volume:
        region[plane.axis] = plane.index
    result = provider.predict_prompted(
        input_image, label, model, hints, plane, full_volume, progress
    ).mask
    if result.shape != input_image.shape[:-1] or not set(np.unique(result)) <= {0, label}:
        raise DomainError("The spatial model returned invalid source geometry or labels.")
    target = mask[tuple(region)]
    prediction = result if crop else result[tuple(region)]
    footprint = region_pixels(crop) if crop else np.ones((1,) * target.ndim, dtype=bool)
    selected = (prediction == label) & footprint
    if np.any(selected & (target != 0) & (target != label)):
        raise Conflict(
            "The prediction overlaps another preserved label. Adjust the prompt "
            "or review the labels separately."
        )
    target[(target == label) & footprint] = 0
    target[selected] = label
    return mask
