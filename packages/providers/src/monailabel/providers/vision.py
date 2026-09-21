"""Structured image requests shared by segmentation, localization and classification."""

import base64
from typing import TYPE_CHECKING, Any, Literal

from monailabel.core.errors import DomainError

if TYPE_CHECKING:
    from monailabel.providers.remote import RemoteConfig

VisionProvider = Literal["openai-polygons", "openai-chat-polygons", "anthropic-polygons"]
VISION_PROVIDERS: tuple[VisionProvider, ...] = (
    "openai-polygons",
    "openai-chat-polygons",
    "anthropic-polygons",
)


class IncompleteVisionResponse(ValueError):
    """The provider exhausted its output budget before completing the annotation."""


def anthropic_schema(value: Any) -> Any:
    """Adapt schema constraints for Messages; callers still validate the original contract."""
    if isinstance(value, list):
        return [anthropic_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: anthropic_schema(item) for key, item in value.items()}
    if isinstance(result.get("type"), str):
        limits = []
        for key in (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
        ):
            if key in result and not (key == "minItems" and result[key] in (0, 1)):
                limits.append(f"{key}: {result.pop(key)}")
        if limits:
            result["description"] = (
                result.get("description", "") + " Constraints: " + ", ".join(limits) + "."
            ).strip()
    return result


def vision_request(
    provider: str,
    config: "RemoteConfig",
    instructions: str,
    text: str,
    images: list[bytes],
    specification: dict[str, Any],
) -> dict[str, Any]:
    if not config.model:
        raise DomainError("The vision provider requires an explicit model ID.")
    encoded = [base64.b64encode(image).decode() for image in images]
    urls = ["data:image/png;base64," + content for content in encoded]
    if provider == "anthropic-polygons":
        return {
            "model": config.model,
            "system": instructions,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}]
                    + [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": content,
                            },
                        }
                        for content in encoded
                    ],
                }
            ],
            "max_tokens": config.max_output_tokens,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": anthropic_schema(specification["schema"]),
                }
            },
        }
    if provider == "openai-chat-polygons":
        request = {
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
            config.max_tokens_field: config.max_output_tokens,
        }
        if config.reasoning_effort is not None:
            request["reasoning_effort"] = config.reasoning_effort
        return request
    if provider != "openai-polygons":
        raise DomainError("Choose a supported vision API provider.")
    request = {
        "model": config.model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
                + [{"type": "input_image", "image_url": url, "detail": "high"} for url in urls],
            }
        ],
        "text": {"format": {"type": "json_schema", **specification}},
        "max_output_tokens": config.max_output_tokens,
        "store": False,
    }
    if config.reasoning_effort is not None:
        request["reasoning"] = {"effort": config.reasoning_effort}
    return request


def vision_text(provider: str, payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Expected a vision response object")
    if provider == "openai-chat-polygons":
        choice = payload["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("Missing annotation message")
        if choice.get("finish_reason") == "length":
            raise IncompleteVisionResponse()
        if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
            raise ValueError("Incomplete or refused annotation")
        text = choice["message"]["content"]
    elif provider == "anthropic-polygons":
        if payload.get("stop_reason") == "max_tokens":
            raise IncompleteVisionResponse()
        if payload.get("stop_reason") != "end_turn":
            raise ValueError("Incomplete or refused annotation")
        blocks = payload["content"]
        if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
            raise ValueError("Invalid annotation content")
        if any(
            block.get("type") not in {"text", "thinking", "redacted_thinking"} for block in blocks
        ):
            raise ValueError("Unexpected annotation response")
        text = "".join(block["text"] for block in blocks if block.get("type") == "text")
    elif provider == "openai-polygons":
        if payload.get("status") == "incomplete":
            raise IncompleteVisionResponse()
        if payload.get("status") not in {None, "completed"}:
            raise ValueError("Incomplete annotation")
        output = payload["output"]
        if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
            raise ValueError("Invalid annotation output")
        blocks = [
            block
            for item in output
            if item.get("type") == "message"
            for block in item.get("content", [])
        ]
        if any(
            not isinstance(block, dict) or block.get("type") != "output_text" for block in blocks
        ):
            raise ValueError("Unexpected or refused annotation response")
        text = "".join(block["text"] for block in blocks)
    else:
        raise ValueError("Unsupported vision provider")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing annotation response")
    return text
