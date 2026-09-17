"""Validate viewer hints and merge promptable-model output into an immutable proposal."""

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.geometry import region_pixels
from monailabel.core.models import AnnotateRequest, Asset, ModelRecord, SpatialPrompt
from monailabel.core.ports import Image, Mask, Progress, PromptedSegmenter


def validate(
    asset: Asset, request: AnnotateRequest, model: ModelRecord, labels: list[int]
) -> SpatialPrompt:
    if len(labels) != 1:
        raise DomainError(
            "SAM annotates one prompted object at a time. Name one target for this box or points."
        )
    if request.image_tiling:
        raise DomainError(
            "SAM needs an object box or points. Select a region; "
            "whole-slide automatic nuclei detection needs a nuclei model."
        )
    if model.provider == "medsam2" and asset.kind != "volume3d":
        raise DomainError("Use SAM 2.1 for a 2D image; MedSAM2 requires a medical volume.")
    if model.provider == "sam2" and request.all_slices:
        raise DomainError(
            "Use MedSAM2 to propagate a prompt through a volume, or SAM 2.1 on one slice."
        )
    if asset.kind == "volume3d" and (request.slice is None or request.slice.window is None):
        raise DomainError(
            "Open a source slice in Slicer or OHIF to supply the SAM seed plane "
            "and intensity window."
        )
    spatial = request.spatial_prompt
    crop = request.image_region
    if spatial is None and crop:
        spatial = SpatialPrompt(
            box=[[crop.y, crop.x], [crop.y + crop.height - 1, crop.x + crop.width - 1]]
        )
    if spatial is None:
        raise DomainError(
            "SAM needs a spatial hint. Select an object box or add positive/negative "
            "points in the viewer, then retry."
        )
    coordinates = [point.coordinates for point in spatial.points] + (spatial.box or [])
    if any(
        len(point) != len(asset.spatial_shape)
        or any(v > size - 1 for v, size in zip(point, asset.spatial_shape, strict=True))
        for point in coordinates
    ):
        raise DomainError("SAM prompts must lie inside the original image in source coordinates.")
    plane = request.slice
    axes = [axis for axis in range(len(asset.spatial_shape)) if plane is None or axis != plane.axis]
    if spatial.box and any(spatial.box[0][axis] >= spatial.box[1][axis] for axis in axes):
        raise DomainError("Draw a SAM box with nonzero width and height.")
    if plane:
        if any(abs(p.coordinates[plane.axis] - plane.index) > 0.5 for p in spatial.points):
            raise DomainError("Place SAM points on the selected source slice.")
        if (
            spatial.box
            and not spatial.box[0][plane.axis] - 0.5
            <= plane.index
            <= spatial.box[1][plane.axis] + 0.5
        ):
            raise DomainError("Select a slice intersecting the SAM box or ROI.")
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
            raise DomainError("SAM hints must lie inside the selected region.")
    elif plane and not full_volume:
        region[plane.axis] = plane.index
    result = provider.predict_prompted(
        input_image, label, model, hints, plane, full_volume, progress
    ).mask
    if result.shape != input_image.shape[:-1] or not set(np.unique(result)) <= {0, label}:
        raise DomainError("SAM returned invalid source geometry or labels.")
    target = mask[tuple(region)]
    prediction = result if crop else result[tuple(region)]
    footprint = region_pixels(crop) if crop else np.ones((1,) * target.ndim, dtype=bool)
    selected = (prediction == label) & footprint
    if np.any(selected & (target != 0) & (target != label)):
        raise Conflict(
            "SAM overlaps another preserved label. Adjust the prompt "
            "or review the labels separately."
        )
    target[(target == label) & footprint] = 0
    target[selected] = label
    return mask
