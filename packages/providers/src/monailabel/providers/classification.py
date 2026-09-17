"""Object classification with replaceable OpenAI-compatible vision endpoints."""

import base64
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
        request_headers = headers(config)
        if config.credential_id:
            if self.credentials is None or model.project_id is None:
                raise DomainError("No credential resolver is available.")
            request_headers = {
                "Authorization": "Bearer "
                + self.credentials(model.project_id, config.credential_id)
            }
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
        urls = [
            "data:image/png;base64," + base64.b64encode(content).decode()
            for content in (png_bytes(image), marked_image(image, objects))
        ]
        specification = {"name": "object_classification", "strict": True, "schema": output}
        chat = model.provider == "openai-chat-polygons"
        if chat:
            body = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                        + [
                            {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
                            for url in urls
                        ],
                    },
                ],
                "response_format": {"type": "json_schema", "json_schema": specification},
                "max_completion_tokens": config.max_output_tokens,
            }
            if config.reasoning_effort is not None:
                body["reasoning_effort"] = config.reasoning_effort
        else:
            body = {
                "model": config.model,
                "instructions": instructions,
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}]
                        + [
                            {"type": "input_image", "image_url": url, "detail": "high"}
                            for url in urls
                        ],
                    }
                ],
                "text": {"format": {"type": "json_schema", **specification}},
                "max_output_tokens": config.max_output_tokens,
            }
            if config.reasoning_effort is not None:
                body["reasoning"] = {"effort": config.reasoning_effort}
        try:
            with httpx.Client(timeout=config.timeout, follow_redirects=False) as client:
                response = client.post(config.url, headers=request_headers, json=body)
                response.raise_for_status()
                payload = response.json()
            if chat:
                choice = payload["choices"][0]
                if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                    raise ValueError("Incomplete classification")
                text = choice["message"]["content"]
            else:
                if payload.get("status") == "incomplete":
                    raise ValueError("Incomplete classification")
                text = "".join(
                    c["text"]
                    for item in payload["output"]
                    if item.get("type") == "message"
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                )
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
