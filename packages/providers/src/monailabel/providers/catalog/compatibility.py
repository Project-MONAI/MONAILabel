"""Conservative compatibility fallback for catalogs without capability metadata.

These are model IDs, not prefix guesses: image/audio/code/research variants must
not inherit a general model's capabilities. Only dated snapshots inherit them.
Sources and the update procedure are documented in docs/providers.md.
"""

import re

from monailabel.providers.catalog.services import ServiceId

KNOWN_MODELS = {
    "openai": frozenset(
        {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
            "gpt-5.6",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-6-astra",
            "o1",
            "o3",
            "o4-mini",
        }
    ),
    "anthropic": frozenset(
        {
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
        }
    ),
    "gemini": frozenset(
        {
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
        }
    ),
}


def known_compatible(service: ServiceId, identifier: str) -> bool:
    model = identifier
    if service == "nvidia":
        # Gateway route prefixes belong to NVIDIA. Only the underlying model
        # identity is matched here; cloud/region/vendor routes are never enumerated.
        model = identifier.rsplit("/", 1)[-1].removeprefix("bedrock-").removesuffix("-v1")
    # This first GPT-4o snapshot predates JSON-schema structured output.
    if model == "gpt-4o-2024-05-13":
        return False
    model = re.sub(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$", "", model)
    if service == "nvidia":
        return any(model in models for models in KNOWN_MODELS.values())
    return model in KNOWN_MODELS.get(service, ())


def compatible(service: ServiceId, identifier: str, row: dict[str, object]) -> bool:
    modes = (None, "chat", "responses") if service == "nvidia" else (None, "chat")
    if row.get("mode") not in modes:
        return False
    if service == "gemini":
        methods = row.get("supportedGenerationMethods")
        if not isinstance(methods, list) or "generateContent" not in methods:
            return False
    image: object = None
    schema: object = None
    if service == "anthropic":
        capabilities = row.get("capabilities")
        if isinstance(capabilities, dict):
            image_info = capabilities.get("image_input")
            schema_info = capabilities.get("structured_outputs")
            image = image_info.get("supported") if isinstance(image_info, dict) else None
            schema = schema_info.get("supported") if isinstance(schema_info, dict) else None
    elif service == "nvidia":
        images = []
        schemas = []
        for metadata in (row, row.get("metadata"), row.get("model_info")):
            if isinstance(metadata, dict):
                images.append(metadata.get("supports_vision"))
                schemas.append(metadata.get("supports_response_schema"))
                if metadata.get("mode") not in modes:
                    return False
        if any(v is not None and v is not True for v in images + schemas):
            return False
        image = True if True in images else None
        schema = True if True in schemas else None
    # Explicitly unsupported or malformed capabilities take precedence over defaults.
    if any(value is not None and value is not True for value in (image, schema)):
        return False
    return (image is True and schema is True) or known_compatible(service, identifier)
