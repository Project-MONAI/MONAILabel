"""Idempotent model presets. Registration never invokes inference or modifies weights."""

import os

from monailabel.core.models import ModelRecord, Project
from monailabel.providers.sam import MODELS as SAM_MODELS
from monailabel.providers.vista3d import targets as vista_targets
from monailabel.server.storage import Store


class Presets:
    def __init__(self, store: Store):
        self.store = store
        self.enabled = os.environ.get("MONAILABEL_PRELOAD_MODELS", "1") != "0"

    def ensure(self, project_id: str) -> None:
        if not self.enabled:
            return
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            if project.is_demo:
                return
            models = session.list(ModelRecord, project_id)
            base = next((m for m in models if m.preset == "vista3d"), None)
            if base is None:
                base = ModelRecord(
                    project_id=project_id,
                    name="VISTA3D",
                    provider="vista3d",
                    label_ids=[0],
                    preset="vista3d",
                    read_only=True,
                )
                session.insert(base)
            elif base.name == "VISTA3D · CT foundation · read-only":
                session.update(base.model_copy(update={"name": "VISTA3D"}))
            for provider, spec in SAM_MODELS.items():
                if not any(model.preset == provider for model in models):
                    session.insert(
                        ModelRecord(
                            project_id=project_id,
                            name=spec.name,
                            provider=provider,
                            label_ids=[0],
                            preset=provider,
                            read_only=True,
                        )
                    )
            for key, name, legacy_name, provider_model in (
                (
                    "nvidia-sol",
                    "GPT-5.6 Sol",
                    "GPT-5.6 Sol · NVIDIA gateway",
                    "switchyard/openai/gpt-5.6-sol",
                ),
                (
                    "nvidia-astra",
                    "GPT-6 Astra",
                    "GPT-6 Astra · NVIDIA gateway (paid)",
                    "azure/openai/gpt-6-astra",
                ),
            ):
                model = next(
                    (
                        m
                        for m in models
                        if m.preset == key
                        or (
                            m.provider == "openai-chat-polygons"
                            and m.config.get("model") == provider_model
                            and m.config.get("url")
                            == "https://inference-api.nvidia.com/v1/chat/completions"
                        )
                    ),
                    None,
                )
                if model is None:
                    model = ModelRecord(
                        project_id=project_id,
                        name=name,
                        provider="openai-chat-polygons",
                        label_ids=[0],
                        preset=key,
                        config={
                            "url": "https://inference-api.nvidia.com/v1/chat/completions",
                            "model": provider_model,
                            "token_env": "NV_INFERENCE_API_KEY",
                            "timeout": 180,
                            "max_output_tokens": 16384,
                            "reasoning_effort": "high",
                        },
                    )
                    session.insert(model)
                elif model.name == legacy_name:
                    # Only replace our old generated title; preserve custom names/configuration.
                    model = model.model_copy(update={"name": name})
                    session.update(model)
            if project.annotation_model_id is None:
                defaults = project.defaults or {
                    label.id: base.id
                    for label in project.labels
                    if label.id and label.name.casefold() in vista_targets()
                }
                session.update(
                    project.model_copy(
                        update={
                            "annotation_model_id": base.id,
                            "defaults": defaults,
                            "version": project.version + 1,
                        }
                    )
                )
