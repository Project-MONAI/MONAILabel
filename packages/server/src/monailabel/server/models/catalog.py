"""Project authorization is handled by routes; secrets remain on the server."""

import os
from collections.abc import Callable

from pydantic import Field, JsonValue

from monailabel.core.errors import DomainError
from monailabel.core.models import Contract, ModelRecord, ModelRegister, Project
from monailabel.providers.catalog import SERVICES, ServiceId, discover
from monailabel.providers.catalog.discovery import CatalogModel
from monailabel.providers.catalog.presets import (
    HOSTED_PRESETS,
    PresetConnection,
    PresetSpec,
    resolve_presets,
)
from monailabel.providers.catalog.selection import recent_models
from monailabel.providers.vision import VisionProvider
from monailabel.server.models.presets import Presets
from monailabel.server.models.service import Models


class CatalogRequest(Contract):
    service: ServiceId
    credential_id: str | None = Field(default=None, min_length=1, max_length=100)


class ImportModelRequest(CatalogRequest):
    model: str = Field(min_length=1, max_length=300)
    name: str = Field(default="", max_length=120)
    max_output_tokens: int = Field(default=4096, ge=64, le=16384)


class ProviderChangeRequest(Contract):
    base_version: int = Field(ge=0)
    connection: ImportModelRequest | None = None


class ServiceInfo(Contract):
    id: ServiceId
    name: str
    token_env: str
    configured: bool


class ModelChoice(Contract):
    id: str
    name: str
    provider: VisionProvider
    url: str
    existing_model_id: str | None = None


class ModelCatalog(Contract):
    models: list[ModelChoice]
    excluded: int


class ModelCatalogs:
    def __init__(self, models: Models, credentials: Callable[[str, str], str], presets: Presets):
        self.models = models
        self.credentials = credentials
        self.presets = presets

    def services(self, project_id: str) -> list[ServiceInfo]:
        self.models.store.get(Project, project_id)
        return [
            ServiceInfo(
                id=s.id,
                name=s.name,
                token_env=s.token_env,
                configured=bool(os.environ.get(s.token_env, "").strip()),
            )
            for s in SERVICES.values()
        ]

    @staticmethod
    def connection(
        request: CatalogRequest, model: CatalogModel | ModelChoice
    ) -> dict[str, JsonValue]:
        service = SERVICES[request.service]
        return {
            "url": model.url,
            **(
                {"credential_id": request.credential_id}
                if request.credential_id
                else {"token_env": service.token_env}
            ),
        }

    def list_models(
        self,
        project_id: str,
        request: CatalogRequest,
        *,
        latest_only: bool = True,
    ) -> ModelCatalog:
        self.models.store.get(Project, project_id)
        service = SERVICES[request.service]
        key = (
            self.credentials(project_id, request.credential_id)
            if request.credential_id
            else os.environ.get(service.token_env, "")
        )
        if not key.strip():
            raise DomainError(f"Set {service.token_env} on the server or choose a saved API key.")
        catalog = discover(service, key.strip())
        if latest_only and service.id == "nvidia":
            catalog = recent_models(catalog)
        existing = self.models.available(project_id)
        choices = []
        for model in sorted(catalog.models, key=lambda m: (m.name.casefold(), m.id)):
            connection = self.connection(request, model)
            previous = next(
                (
                    old
                    for old in existing
                    if old.provider == model.provider
                    and old.config.get("model") == model.id
                    and all(old.config.get(k) == v for k, v in connection.items())
                ),
                None,
            )
            choices.append(
                ModelChoice(
                    id=model.id,
                    name=model.name,
                    provider=model.provider,
                    url=model.url,
                    existing_model_id=previous.id if previous else None,
                )
            )
        return ModelCatalog(models=choices, excluded=catalog.excluded)

    def import_model(self, project_id: str, request: ImportModelRequest) -> ModelRecord:
        # Re-read the account's catalog so a changed key, revoked model or forged UI
        # selection cannot import an unavailable/incompatible catalog entry.
        catalog = self.list_models(project_id, request)
        choice = next((m for m in catalog.models if m.id == request.model), None)
        if choice is None:
            raise DomainError("This model is not available for annotation. Refresh the model list.")
        if choice.existing_model_id:
            return self.models.get(project_id, choice.existing_model_id)
        return self.models.register(
            project_id,
            ModelRegister(
                name=request.name.strip() or choice.name[:120],
                provider=choice.provider,
                label_ids=[0],
                config=self.inference_config(request, choice),
            ),
        )

    @classmethod
    def inference_config(
        cls, request: ImportModelRequest, model: ModelChoice
    ) -> dict[str, JsonValue]:
        service = SERVICES[request.service]
        config = cls.connection(request, model) | {
            "model": request.model,
            "timeout": 180,
            "max_output_tokens": request.max_output_tokens,
        }
        if model.provider == "openai-chat-polygons":
            config["max_tokens_field"] = service.max_tokens_field
        return config

    def preset(self, project_id: str, model_id: str) -> PresetSpec:
        model = self.models.get(project_id, model_id)
        spec = next((s for s in HOSTED_PRESETS if s.key == model.preset), None)
        if spec is None:
            raise DomainError("Provider switching is available for hosted presets.")
        return spec

    def preset_services(self, project_id: str, model_id: str) -> list[ServiceInfo]:
        spec = self.preset(project_id, model_id)
        return [s for s in self.services(project_id) if s.id in {"nvidia", spec.direct_service}]

    def preset_catalog(
        self, project_id: str, model_id: str, request: CatalogRequest
    ) -> ModelCatalog:
        spec = self.preset(project_id, model_id)
        if request.service not in {"nvidia", spec.direct_service}:
            raise DomainError("This provider does not host the selected preset.")
        catalog = self.list_models(project_id, request, latest_only=False)
        # An existing connection can still be selected to pin an automatic preset.
        choices = [
            m.model_copy(update={"existing_model_id": None})
            for m in catalog.models
            if (
                spec.matches_gateway(m.id)
                if request.service == "nvidia"
                else m.id == spec.direct_model
            )
        ]
        return ModelCatalog(
            models=choices, excluded=catalog.excluded + len(catalog.models) - len(choices)
        )

    def change_provider(
        self, project_id: str, model_id: str, request: ProviderChangeRequest
    ) -> ModelRecord:
        spec = self.preset(project_id, model_id)
        selected: PresetConnection | None
        if request.connection:
            catalog = self.preset_catalog(project_id, model_id, request.connection)
            choice = next((m for m in catalog.models if m.id == request.connection.model), None)
            if choice is None:
                raise DomainError("This provider does not offer the selected preset with this key.")
            config = self.inference_config(request.connection, choice)
            if spec.reasoning_effort:
                config["reasoning_effort"] = spec.reasoning_effort
            selected = PresetConnection(choice.provider, config)
        else:
            selected = resolve_presets((spec,)).get(spec.key)
            if selected is None:
                raise DomainError(
                    "No configured provider currently offers this preset. "
                    "Your connection is unchanged."
                )
        return self.presets.change_connection(
            project_id,
            model_id,
            request.base_version,
            selected,
            automatic=request.connection is None,
        )
