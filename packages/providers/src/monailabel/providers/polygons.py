"""Shared 2D polygon contract for vision models using different HTTP APIs."""

import json
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import Field, ValidationError

from monailabel.core.models import Contract, Label
from monailabel.core.ports import Image, Mask

INSTRUCTIONS = (
    "Create candidate segmentation polygons for review. Use the supplied foreground class IDs. "
    "Coordinates are input-image pixel x,y, origin top-left, with continuous boundary coordinates "
    "0 <= x <= width and 0 <= y <= height. The right and bottom image edges are width and height. "
    "Trace the visible part of each object, including objects cut by the image boundary; "
    "do not draw a bounding box. Polygons of different classes must not overlap. "
    "Return an empty polygons list if no target is visible or you cannot locate it. "
    "The image may be rotated or reflected relative to conventional clinical displays."
)


class Point(Contract):
    x: float = Field(strict=True, allow_inf_nan=False, ge=0)
    y: float = Field(strict=True, allow_inf_nan=False, ge=0)


class Polygon(Contract):
    label_id: int = Field(strict=True, ge=1, le=255)
    points: list[Point] = Field(min_length=3, max_length=512)


class Polygons(Contract):
    polygons: list[Polygon] = Field(max_length=64)


class PolygonOutputError(ValueError):
    """A diagnostic safe to display, without echoing arbitrary provider content."""


def schema(labels: list[Label], shape: tuple[int, ...]) -> dict[str, Any]:
    result = Polygons.model_json_schema()
    result["$defs"]["Polygon"]["properties"]["label_id"]["enum"] = [x.id for x in labels if x.id]
    coordinates = result["$defs"]["Point"]["properties"]
    coordinates["x"]["maximum"] = shape[1]
    coordinates["y"]["maximum"] = shape[0]
    return result


def image_prompt(image: Image, labels: list[Label], prompt: str) -> str:
    return json.dumps(
        {
            "width": image.shape[1],
            "height": image.shape[0],
            "labels": [x.model_dump() for x in labels],
            "request": prompt,
        }
    )


def mask_from_polygons(payload: object, shape: tuple[int, ...], label_ids: list[int]) -> Mask:
    try:
        parsed = Polygons.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        if any(error["loc"] == ("polygons",) and error["type"] == "too_long" for error in errors):
            raise PolygonOutputError("The response exceeds the 64-polygon limit.") from exc
        raise PolygonOutputError(
            "Invalid polygon fields: expected foreground label IDs and 3–512 numeric x/y "
            "points per polygon."
        ) from exc
    mask = np.zeros(shape, dtype=np.uint8)
    for index, polygon in enumerate(parsed.polygons, start=1):
        label = polygon.label_id
        if label not in label_ids:
            raise PolygonOutputError(f"Polygon {index} uses an unregistered foreground label.")
        vertices = []
        for point in polygon.points:
            # Polygon vertices describe continuous boundaries, not array indices.
            # Pillow rasterizes into the fixed-size canvas, clipping boundary vertices.
            if not (0 <= point.x <= shape[1] and 0 <= point.y <= shape[0]):
                raise PolygonOutputError(
                    f"Polygon {index} has coordinates outside the {shape[1]} × {shape[0]} "
                    f"input image; expected x from 0 to {shape[1]} and y from 0 to {shape[0]}."
                )
            vertices.append((point.x, point.y))
        canvas = PILImage.new("L", (shape[1], shape[0]), 0)
        ImageDraw.Draw(canvas).polygon(vertices, fill=1)
        selected = np.asarray(canvas) > 0
        if np.any(selected & (mask != 0) & (mask != label)):
            raise PolygonOutputError(
                f"Polygon {index} overlaps a different class. This adapter uses exclusive labels."
            )
        mask[selected] = label
    return mask
