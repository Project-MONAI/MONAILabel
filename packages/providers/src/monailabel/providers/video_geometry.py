"""Source-grid masks and reviewable polygon approximations for video providers."""

import io

import cv2
import numpy as np
from PIL import Image as PILImage

from monailabel.core.errors import DomainError
from monailabel.core.ports import Mask
from monailabel.core.video import PolygonKeyframe, TrackKeyframe


def box_from_mask(mask: Mask, frame: int) -> TrackKeyframe:
    count, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if count != 2:
        raise DomainError(
            "The segmentation must contain one connected tool. "
            "Use a more specific prompt or select a starting annotation."
        )
    ys, xs = np.where(mask)
    return TrackKeyframe(
        frame=frame,
        box=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
    )


def mask_png(mask: Mask) -> bytes:
    stream = io.BytesIO()
    PILImage.fromarray((mask > 0).astype(np.uint8) * 255).convert("1").save(stream, format="PNG")
    return stream.getvalue()


def polygon_from_mask(
    mask: Mask, frame: int, occluded: bool = False
) -> tuple[PolygonKeyframe, bool]:
    # Trace pixel-cell edges, including pixels touching the image border.
    raster = np.zeros((mask.shape[0] * 2 + 1, mask.shape[1] * 2 + 1), dtype=np.uint8)
    raster[1::2, 1::2] = mask > 0
    raster = cv2.dilate(raster, np.ones((3, 3), dtype=np.uint8))
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise DomainError(
            "The model did not segment a visible tool. Choose another frame or prompt."
        )
    outer = [c for c, h in zip(contours, hierarchy[0], strict=True) if h[3] == -1]
    contour = max(outer, key=cv2.contourArea)
    points = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2).astype(np.float64) / 2
    if len(points) < 3 or len(points) > 2048:
        raise DomainError(
            "This segmentation cannot be represented by a usable polygon. Refine the target."
        )
    # A stable starting vertex/orientation makes later manual interpolation predictable.
    if cv2.contourArea(points.astype(np.float32), oriented=True) < 0:
        points = points[::-1]
    start = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    points = np.roll(points, -start, axis=0)
    return PolygonKeyframe(frame=frame, points=points.ravel().tolist(), occluded=occluded), len(
        contours
    ) > 1
