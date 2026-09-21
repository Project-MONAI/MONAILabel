"""Object classification with replaceable structured vision endpoints."""

import io
import json
from collections.abc import Callable

import httpx
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFilter
from pydantic import Field

from monailabel.core.errors import DomainError
from monailabel.core.geometry import region_pixels
from monailabel.core.models import ClassificationObject, Contract, ModelRecord, ObjectClassification
from monailabel.core.ports import Image
from monailabel.providers.remote import headers, parse_config, png_bytes
from monailabel.providers.vision import vision_request, vision_text


class ClassifiedMarker(Contract):
    marker: int = Field(strict=True, ge=1, le=128)
    category: str | None


class ClassifiedMarkers(Contract):
    classifications: list[ClassifiedMarker] = Field(min_length=1, max_length=128)


def marked_image(image: Image, objects: list[ClassificationObject]) -> bytes:
    canvas = PILImage.open(io.BytesIO(png_bytes(image))).convert("RGB")
    for number, item in enumerate(objects, start=1):
        r = item.region
        mask = PILImage.fromarray(region_pixels(r).astype(np.uint8) * 255)
        # Pad before finding edges so a full rectangular footprint still has an outline.
        padded = PILImage.new("L", (r.width + 2, r.height + 2))
        padded.paste(mask, (1, 1))
        outline = padded.filter(ImageFilter.FIND_EDGES).crop((1, 1, r.width + 1, r.height + 1))
        canvas.paste((0, 255, 0), (r.x, r.y, r.x + r.width, r.y + r.height), outline)
        draw = ImageDraw.Draw(canvas)
        position = (r.x, r.y)
        draw.rectangle(draw.textbbox(position, str(number)), fill="black")
        draw.text(position, str(number), fill="yellow")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()


class RemoteClassifier:
    def __init__(self, credentials: Callable[[str, str], str] | None = None):
        self.credentials = credentials

    def classify(
        self,
        image: Image,
        objects: list[ClassificationObject],
        categories: list[str],
        prompt: str,
        model: ModelRecord,
    ) -> list[ObjectClassification]:
        config = parse_config(model)
        request_headers = headers(config, model, self.credentials)
        if not config.model:
            raise DomainError("Choose an explicit vision model for object classification.")
        output = ClassifiedMarkers.model_json_schema()
        definition = output["$defs"]["ClassifiedMarker"]["properties"]
        definition["marker"]["enum"] = list(range(1, len(objects) + 1))
        definition["category"] = {
            "anyOf": [{"type": "string", "enum": categories}, {"type": "null"}]
        }
        instructions = (
            "Classify the marked annotation objects into the supplied categories for human review. "
            "Image 1 is the original image; image 2 has numbered object outlines. "
            "Return one classification for EVERY marker, preserving its number. "
            "Use category null when uncertain, ambiguous, not visible, "
            "or unsupported by the image. "
            "Do not infer molecular status or staining positivity without evidence. "
            "Classifications are candidates, not diagnoses. Do not change object boundaries."
        )
        text = json.dumps(
            {
                "request": prompt,
                "categories": categories,
                "objects": [
                    {
                        "marker": i,
                        "bounds": [o.region.x, o.region.y, o.region.width, o.region.height],
                    }
                    for i, o in enumerate(objects, start=1)
                ],
            }
        )
        body = vision_request(
            model.provider,
            config,
            instructions,
            text,
            [png_bytes(image), marked_image(image, objects)],
            {"name": "object_classification", "strict": True, "schema": output},
        )
        try:
            with httpx.Client(timeout=config.timeout, follow_redirects=False) as client:
                response = client.post(config.url, headers=request_headers, json=body)
                response.raise_for_status()
                payload = response.json()
            text = vision_text(model.provider, payload)
            result = ClassifiedMarkers.model_validate_json(text)
            numbers = [item.marker for item in result.classifications]
            if sorted(numbers) != list(range(1, len(objects) + 1)):
                raise ValueError("Missing or duplicate object markers")
            if any(
                item.category is not None and item.category not in categories
                for item in result.classifications
            ):
                raise ValueError("Unknown category")
            return [
                ObjectClassification(object_id=objects[item.marker - 1].id, category=item.category)
                for item in result.classifications
            ]
        except httpx.HTTPStatusError as exc:
            raise DomainError(
                f"Classification provider returned HTTP {exc.response.status_code}.",
                code="provider_error",
                status=502,
            ) from exc
        except httpx.RequestError as exc:
            raise DomainError(
                "Classification provider request failed.", code="provider_error", status=502
            ) from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise DomainError(
                "Classification response must contain exactly one valid category (or abstention) "
                "per object. No classifications were applied. "
                "Check the output limit if the response was incomplete.",
                code="provider_output_invalid",
                status=502,
            ) from exc
