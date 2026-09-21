"""Explicit HTTP contracts. Credentials are resolved from environment references."""

import base64
import io
import json
import os
from collections.abc import Callable
from typing import Literal
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image as PILImage
from pydantic import Field, ValidationError

from monailabel.core.errors import DomainError
from monailabel.core.models import Contract, Label, ModelRecord
from monailabel.core.ports import Image, Prediction
from monailabel.providers.polygons import (
    INSTRUCTIONS,
    PolygonOutputError,
    image_prompt,
    mask_from_polygons,
    schema,
)
from monailabel.providers.vision import (
    IncompleteVisionResponse,
    VisionProvider,
    vision_request,
    vision_text,
)


class RemoteConfig(Contract):
    url: str
    token_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    credential_id: str | None = None
    timeout: float = Field(default=120, ge=1, le=600)
    model: str | None = None
    max_output_tokens: int = Field(default=4096, ge=64, le=16384)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    label_map: dict[str, int] = Field(default_factory=dict)


def parse_config(model: ModelRecord) -> RemoteConfig:
    try:
        config = RemoteConfig.model_validate(model.config)
    except ValidationError as exc:
        raise DomainError("Invalid provider configuration; see the provider contract.") from exc
    url = urlparse(config.url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise DomainError("Provider URL must be HTTP(S), with credentials supplied by token_env.")
    if url.query or url.fragment:
        raise DomainError("Provider URLs cannot contain query strings or fragments.")
    if config.credential_id and config.token_env:
        raise DomainError("Choose a saved credential or an environment reference, not both.")
    if model.provider == "anthropic-polygons" and config.reasoning_effort is not None:
        raise DomainError("Anthropic annotation does not accept GPT reasoning_effort settings.")
    return config


def headers(
    config: RemoteConfig,
    model: ModelRecord,
    credentials: Callable[[str, str], str] | None = None,
) -> dict[str, str]:
    token = None
    if config.token_env:
        token = os.environ.get(config.token_env)
        if not token:
            raise DomainError(f"Set {config.token_env} on the server before using this provider.")
    elif config.credential_id:
        if credentials is None or model.project_id is None:
            raise DomainError("No credential resolver is available.")
        token = credentials(model.project_id, config.credential_id)
    if model.provider == "anthropic-polygons":
        return {"anthropic-version": "2023-06-01", **({"x-api-key": token} if token else {})}
    return {"Authorization": f"Bearer {token}"} if token else {}


def png_bytes(image: Image) -> bytes:
    if image.ndim != 3 or image.shape[-1] not in {1, 3}:
        raise DomainError("This provider accepts 2D images only; volume slicing is not implicit.")
    values = np.clip(image * 255, 0, 255).astype(np.uint8)
    if values.shape[-1] == 1:
        values = values[..., 0]
    stream = io.BytesIO()
    PILImage.fromarray(values).save(stream, format="PNG")
    return stream.getvalue()


class RemoteSegmenter:
    def __init__(
        self,
        kind: Literal["http-mask", "huggingface"] | VisionProvider,
        credentials: Callable[[str, str], str] | None = None,
    ):
        self.kind = kind
        self.credentials = credentials

    def predict(
        self, image: Image, labels: list[Label], prompt: str, model: ModelRecord
    ) -> Prediction:
        config = parse_config(model)
        request_headers = headers(config, model, self.credentials)
        try:
            with httpx.Client(timeout=config.timeout, follow_redirects=False) as client:
                if self.kind == "http-mask":
                    response = client.post(
                        config.url,
                        headers=request_headers,
                        json={
                            "image": image.tolist(),
                            "spatial_shape": list(image.shape[:-1]),
                            "labels": [label.model_dump() for label in labels],
                            "prompt": prompt,
                            "model": config.model,
                        },
                    )
                elif self.kind == "huggingface":
                    response = client.post(
                        config.url,
                        headers={**request_headers, "Content-Type": "image/png"},
                        content=png_bytes(image),
                    )
                else:
                    response = client.post(
                        config.url,
                        headers=request_headers,
                        json=vision_request(
                            self.kind,
                            config,
                            INSTRUCTIONS,
                            image_prompt(image, labels, prompt),
                            [png_bytes(image)],
                            {
                                "name": "segmentation",
                                "strict": True,
                                "schema": schema(labels, image.shape[:2]),
                            },
                        ),
                    )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            # Remote bodies and URLs can contain credentials; do not echo them.
            raise DomainError(
                f"Provider returned HTTP {exc.response.status_code}.",
                code="provider_error",
                status=502,
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise DomainError(
                "Provider request failed.", code="provider_error", status=502
            ) from exc

        try:
            if self.kind == "http-mask":
                mask = np.asarray(payload["mask"])
                if not np.issubdtype(mask.dtype, np.integer):
                    raise ValueError("Mask values must be integers.")
            elif self.kind == "huggingface":
                mask = np.zeros(image.shape[:-1], dtype=np.uint8)
                for item in payload:
                    label_id = config.label_map.get(item["label"])
                    if label_id is None:
                        raise ValueError("Every provider label needs an explicit label_map entry.")
                    with PILImage.open(
                        io.BytesIO(base64.b64decode(item["mask"], validate=True))
                    ) as im:
                        if im.size != (mask.shape[1], mask.shape[0]):
                            raise ValueError("Provider mask dimensions do not match the input.")
                        binary = np.asarray(im.convert("L")) > 0
                    if binary.shape != mask.shape:
                        raise ValueError("Provider mask dimensions do not match the input.")
                    if np.any(binary & (mask != 0) & (mask != label_id)):
                        raise ValueError("Provider returned overlapping classes.")
                    mask[binary] = label_id
            else:
                text = vision_text(self.kind, payload)
                mask = mask_from_polygons(json.loads(text), image.shape[:2], model.label_ids)
            if mask.shape != image.shape[:-1] or not set(np.unique(mask)) <= set(model.label_ids):
                raise ValueError("Provider mask shape or class IDs are invalid.")
            return Prediction(mask.astype(np.uint8))
        except IncompleteVisionResponse as exc:
            raise DomainError(
                "Provider output does not satisfy the mask contract: output limit reached or "
                "incomplete response. Increase max_output_tokens in the model configuration "
                "or annotate a smaller image. No partial mask was applied.",
                code="provider_output_truncated",
                status=502,
            ) from exc
        except (PolygonOutputError, json.JSONDecodeError) as exc:
            detail = (
                str(exc) if isinstance(exc, PolygonOutputError) else "Response is not valid JSON."
            )
            raise DomainError(
                "Provider output does not satisfy the mask contract: "
                + detail
                + " No partial mask was applied.",
                code="provider_output_invalid",
                status=502,
            ) from exc
        except (
            KeyError,
            IndexError,
            ValueError,
            TypeError,
            OSError,
            PILImage.DecompressionBombError,
        ) as exc:
            raise DomainError(
                "Provider output does not satisfy the registered mask contract.",
                code="provider_output_invalid",
                status=502,
            ) from exc
