"""Resolve predefined annotation connections once, before any annotation runs."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import JsonValue

from monailabel.core.errors import DomainError
from monailabel.providers.catalog.discovery import CatalogModel, discover
from monailabel.providers.catalog.services import SERVICES, HostedService, ServiceId
from monailabel.providers.vision import VisionProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresetSpec:
    key: str
    name: str
    direct_service: ServiceId
    direct_model: str
    reasoning_effort: str | None = None
    legacy_name: str | None = None

    def matches_gateway(self, identifier: str) -> bool:
        model = identifier.rsplit("/", 1)[-1].removeprefix("bedrock-").removesuffix("-v1")
        return model == self.direct_model


# Existing preset keys are stable identities, including when the route changes.
DEFAULT_HOSTED_PRESET = "nvidia-astra"
HOSTED_PRESETS = (
    PresetSpec(
        "nvidia-astra",
        "GPT-6 Astra",
        "openai",
        "gpt-6-astra",
        "high",
        "GPT-6 Astra · NVIDIA gateway (paid)",
    ),
    PresetSpec(
        "nvidia-claude-opus-5",
        "Claude Opus 5",
        "anthropic",
        "claude-opus-5",
    ),
    PresetSpec(
        "gemini-flash",
        "Gemini 3.5 Flash",
        "gemini",
        "gemini-3.5-flash",
    ),
)


@dataclass(frozen=True)
class PresetConnection:
    provider: VisionProvider
    config: dict[str, JsonValue]


def _available(identifier: ServiceId) -> dict[str, CatalogModel]:
    service = SERVICES[identifier]
    key = os.environ.get(service.token_env, "").strip()
    if not key:
        return {}
    try:
        return {model.id: model for model in discover(service, key).models}
    except DomainError as exc:
        log.warning("Could not discover %s annotation presets: %s", service.name, exc)
        return {}


def connection(service: HostedService, model: CatalogModel) -> PresetConnection:
    config: dict[str, JsonValue] = {
        "url": model.url,
        "model": model.id,
        "token_env": service.token_env,
        "timeout": 180,
        "max_output_tokens": 16384,
    }
    if model.provider == "openai-chat-polygons":
        config["max_tokens_field"] = service.max_tokens_field
    return PresetConnection(model.provider, config)


def resolve_presets(
    specs: tuple[PresetSpec, ...] = HOSTED_PRESETS,
) -> dict[str, PresetConnection]:
    gateway = _available("nvidia")
    routes = {
        spec.key: spec.direct_model
        if spec.direct_model in gateway
        else next((m for m in gateway if spec.matches_gateway(m)), None)
        for spec in specs
    }
    missing = {s.direct_service for s in specs if routes[s.key] is None}
    with ThreadPoolExecutor(max_workers=3) as pool:
        direct = dict(zip(sorted(missing), pool.map(_available, sorted(missing)), strict=True))
    resolved = {}
    for spec in specs:
        model = routes[spec.key]
        if model:
            selected = connection(SERVICES["nvidia"], gateway[model])
        elif spec.direct_model in direct.get(spec.direct_service, {}):
            selected = connection(
                SERVICES[spec.direct_service], direct[spec.direct_service][spec.direct_model]
            )
        else:
            continue
        if spec.reasoning_effort:
            selected.config["reasoning_effort"] = spec.reasoning_effort
        resolved[spec.key] = selected
    return resolved
