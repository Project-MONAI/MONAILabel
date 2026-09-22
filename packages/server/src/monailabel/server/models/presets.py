"""Idempotent model presets. Registration never invokes inference or modifies weights."""

import os
from collections.abc import Callable

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import ModelRecord, Project
from monailabel.providers.catalog import SERVICES, discover
from monailabel.providers.catalog.presets import (
    DEFAULT_HOSTED_PRESET,
    HOSTED_PRESETS,
    PresetConnection,
    resolve_presets,
)
from monailabel.providers.sam import MODELS as SAM_MODELS
from monailabel.providers.vista3d import targets as vista_targets
from monailabel.server.deletion import Deletion
from monailabel.server.storage import Session, Store


class Presets:
    def __init__(self, store: Store, credentials: Callable[[str, str], str]):
        self.store = store
        self.credentials = credentials
        self.enabled = os.environ.get("MONAILABEL_PRELOAD_MODELS", "1") != "0"
        self.hosted = resolve_presets() if self.enabled else {}
        self.manual_availability: dict[str, bool] = {}

    def ensure(self, project_id: str) -> None:
        if not self.enabled:
            return
        latest = {m.preset: m for m in self.store.list(ModelRecord, project_id) if m.preset}
        # Resolve credentials and network metadata before entering a store transaction.
        for model in latest.values():
            if (
                model.preset in {spec.key for spec in HOSTED_PRESETS}
                and model.connection_mode == "manual"
                and model.id not in self.manual_availability
            ):
                self.manual_availability[model.id] = self.manual_available(model)
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            if project.is_demo:
                return
            models = session.list(ModelRecord, project_id)
            base = next((m for m in models if m.preset == "vista3d"), None)
            initialize_defaults = base is None and project.annotation_model_id is None
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
            for provider, sam in SAM_MODELS.items():
                if not any(model.preset == provider for model in models):
                    session.insert(
                        ModelRecord(
                            project_id=project_id,
                            name=sam.name,
                            provider=provider,
                            label_ids=[0],
                            preset=provider,
                            read_only=True,
                        )
                    )
            for spec in HOSTED_PRESETS:
                active = [m for m in models if m.preset == spec.key and not m.archived]
                current = active[-1] if active else None
                previous_choice = latest.get(spec.key)
                if previous_choice and previous_choice.connection_mode == "manual":
                    if self.manual_availability.get(previous_choice.id):
                        if previous_choice.archived:
                            session.update(
                                previous_choice.model_copy(
                                    update={
                                        "archived": False,
                                        "version": previous_choice.version + 1,
                                    }
                                )
                            )
                    elif current:
                        self.retire(session, current, None)
                    continue
                selected = self.hosted.get(spec.key)
                if selected and current and self.matches(current, selected):
                    changes: dict[str, object] = {}
                    if current.name == spec.legacy_name:
                        changes["name"] = spec.name
                    if current.connection_mode is None:
                        changes["connection_mode"] = "automatic"
                    if changes:
                        session.update(current.model_copy(update=changes))
                    continue
                replacement = None
                if selected:
                    replacement = ModelRecord(
                        project_id=project_id,
                        name=spec.name,
                        provider=selected.provider,
                        label_ids=[0],
                        preset=spec.key,
                        config=selected.config,
                        connection_mode="automatic",
                    )
                    session.insert(replacement)
                for previous in active:
                    self.retire(session, previous, replacement)
            standard = next(
                (
                    m
                    for m in session.list(ModelRecord, project_id)
                    if m.preset == DEFAULT_HOSTED_PRESET and not m.archived
                ),
                None,
            )
            for previous in models:
                if previous.preset == "nvidia-sol" and not previous.archived:
                    self.retire(session, previous, standard)
            project = session.get(Project, project_id)
            if project.annotation_model_id is None and initialize_defaults:
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

    @staticmethod
    def matches(model: ModelRecord, selected: PresetConnection) -> bool:
        return model.provider == selected.provider and all(
            model.config.get(key) == selected.config.get(key)
            for key in ("url", "model", "token_env", "credential_id")
        )

    def manual_available(self, model: ModelRecord) -> bool:
        service = next(
            (
                s
                for s in SERVICES.values()
                if model.config.get("url") in (s.inference_url, s.responses_url)
            ),
            None,
        )
        if service is None:
            return False
        try:
            credential = model.config.get("credential_id")
            key = (
                self.credentials(str(model.project_id), str(credential))
                if credential
                else os.environ.get(str(model.config.get("token_env", "")), "")
            )
            return bool(key.strip()) and model.config.get("model") in {
                m.id for m in discover(service, key.strip()).models
            }
        except DomainError:
            return False

    @staticmethod
    def retire(session: Session, previous: ModelRecord, replacement: ModelRecord | None) -> None:
        # Keep the old connection attached to historical proposals and annotations.
        session.update(
            previous.model_copy(update={"archived": True, "version": previous.version + 1})
        )
        project = session.get(Project, str(previous.project_id))
        defaults = {
            label: replacement.id if model_id == previous.id and replacement else model_id
            for label, model_id in project.defaults.items()
            if model_id != previous.id or replacement
        }
        annotation_model_id = project.annotation_model_id
        if annotation_model_id == previous.id:
            annotation_model_id = replacement.id if replacement else None
        if defaults != project.defaults or annotation_model_id != project.annotation_model_id:
            session.update(
                project.model_copy(
                    update={
                        "defaults": defaults,
                        "annotation_model_id": annotation_model_id,
                        "version": project.version + 1,
                    }
                )
            )

    def change_connection(
        self,
        project_id: str,
        model_id: str,
        base_version: int,
        selected: PresetConnection,
        *,
        automatic: bool,
    ) -> ModelRecord:
        with self.store.transaction() as session:
            previous = session.get(ModelRecord, model_id)
            if previous.project_id != project_id:
                raise DomainError("Model belongs to another project.", status=403)
            if previous.archived or previous.version != base_version:
                raise Conflict("The model connection changed. Refresh and try again.")
            if previous.preset not in {spec.key for spec in HOSTED_PRESETS}:
                raise DomainError("Provider switching is available for hosted presets.")
            Deletion.idle(session, project_id)
            mode = "automatic" if automatic else "manual"
            if (
                previous.provider == selected.provider
                and previous.config == selected.config
                and previous.connection_mode == mode
            ):
                return previous
            replacement = ModelRecord(
                project_id=project_id,
                name=previous.name,
                provider=selected.provider,
                label_ids=previous.label_ids,
                preset=previous.preset,
                config=selected.config,
                connection_mode=mode,
            )
            session.insert(replacement)
            self.retire(session, previous, replacement)
        if not automatic:
            self.manual_availability[replacement.id] = True
        return replacement
