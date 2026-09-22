"""Provider registry and validated execution; explicit model choices never fall back."""

import os
from collections.abc import Callable

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
    Asset,
    DeleteModelRequest,
    Learner,
    ModelRecord,
    ModelRegister,
    ModelUpdate,
    Project,
)
from monailabel.core.ports import (
    Classifier,
    Image,
    Mask,
    PromptedSegmenter,
    Segmenter,
    Volume,
    VolumeSegmenter,
)
from monailabel.core.video import VideoAsset
from monailabel.providers.catalog.presets import DEFAULT_HOSTED_PRESET, HOSTED_PRESETS
from monailabel.providers.classification import RemoteClassifier
from monailabel.providers.local import ThresholdSegmenter
from monailabel.providers.remote import RemoteSegmenter, parse_config
from monailabel.providers.sam import MODELS as SAM_MODELS
from monailabel.providers.vision import VISION_PROVIDERS
from monailabel.providers.vista3d import targets as vista_targets
from monailabel.server.deletion import Deletion
from monailabel.server.recipes import Recipes
from monailabel.server.storage import Artifacts, Store


class Models:
    def __init__(self, store: Store, artifacts: Artifacts, credentials: Callable[[str, str], str]):
        self.store, self.artifacts = store, artifacts
        self.credentials = credentials
        self.recipes = Recipes(artifacts)
        self.providers: dict[str, Segmenter] = {
            "threshold": ThresholdSegmenter(),
            "http-mask": RemoteSegmenter("http-mask", credentials),
            "huggingface": RemoteSegmenter("huggingface", credentials),
            **{name: RemoteSegmenter(name, credentials) for name in VISION_PROVIDERS},
        }
        self.classifiers: dict[str, Classifier] = {
            name: RemoteClassifier(credentials) for name in VISION_PROVIDERS
        }

    def get(self, project_id: str, identifier: str) -> ModelRecord:
        model = self.store.get(ModelRecord, identifier)
        if model.project_id != project_id:
            raise DomainError("Model belongs to another project.", status=403)
        if model.archived:
            raise DomainError("This model has been deleted from the active catalog.")
        return model

    @classmethod
    def compatible(cls, model: ModelRecord, asset: Asset | VideoAsset) -> bool:
        if asset.kind == "video" and cls.requires_spatial(model):
            return False  # Video localization needs an automatic seed, not spatial prompts.
        return asset.kind == "volume3d" or not (
            cls.requires_3d(model) or model.provider == "medsam2"
        )

    def available(self, project_id: str, asset_id: str | None = None) -> list[ModelRecord]:
        asset = self.store.get(Asset, asset_id) if asset_id else None
        if asset and asset.project_id != project_id:
            raise DomainError("Selected asset belongs to another project.")
        return [
            model
            for model in self.store.list(ModelRecord, project_id)
            if not model.archived and (asset is None or self.compatible(model, asset))
        ]

    def configured(self, model: ModelRecord) -> bool:
        if model.provider in {recipe.id for recipe in self.recipes.list()}:
            return bool(model.state_key) or (model.provider == "vista3d" and model.read_only)
        if model.provider not in {
            "http-mask",
            "huggingface",
            *VISION_PROVIDERS,
        }:
            return model.provider in self.providers or self.requires_spatial(model)
        try:
            config = parse_config(model)
            if model.provider in VISION_PROVIDERS and not config.model:
                return False
            if config.token_env and not os.environ.get(config.token_env):
                return False
            if config.credential_id:
                self.credentials(str(model.project_id), config.credential_id)
        except DomainError:
            return False
        return True

    def select_for_targets(
        self, project: Project, asset: Asset | VideoAsset, names: list[str], selected: str | None
    ) -> str | None:
        """Honor explicit choices; otherwise match source geometry, targets and defaults."""
        if selected:
            model = self.get(project.id, selected)
            if not self.compatible(model, asset):
                if asset.kind == "video":
                    raise DomainError(f"{model.name} cannot automatically annotate a video frame.")
                raise DomainError(f"{model.name} requires a volume. Choose a model for 2D images.")
            return model.id
        targets = {name.strip().casefold() for name in names}
        labels = {label.id: label.name.casefold() for label in project.labels if label.id}

        def supports(model: ModelRecord, requested: set[str]) -> bool:
            if model.provider == "vista3d" and (model.read_only or model.inherit_targets):
                return requested <= set(vista_targets())
            return self.promptable(model) or requested <= {
                labels[identifier] for identifier in model.label_ids if identifier in labels
            }

        target_defaults = {
            name: project.defaults[identifier]
            for identifier, name in labels.items()
            if name in targets and identifier in project.defaults
        }
        if len(set(target_defaults.values())) > 1 and set(target_defaults) == targets:
            assignments = [
                (name, self.get(project.id, mid)) for name, mid in target_defaults.items()
            ]
            if all(
                self.compatible(model, asset) and supports(model, {name})
                for name, model in assignments
            ):
                return None  # Preserve separate, explicit defaults for each structure.
        available = [model for model in self.available(project.id) if self.compatible(model, asset)]
        candidates = [model for model in available if supports(model, targets)]
        by_id = {model.id: model for model in candidates}
        if len(set(target_defaults.values())) == 1 and set(target_defaults) == targets:
            default = next(iter(target_defaults.values()))
            if default in by_id:
                return default
        if project.annotation_model_id in by_id:
            return project.annotation_model_id
        if not targets:
            return None

        candidates = [model for model in candidates if self.configured(model)]
        specialized = [model for model in candidates if not self.promptable(model)]
        if len(specialized) == 1:
            return specialized[0].id
        if not specialized:
            defaults = (
                ("vista3d", DEFAULT_HOSTED_PRESET)
                if asset.kind == "volume3d"
                else (DEFAULT_HOSTED_PRESET,)
            )
            for preset in defaults:
                standard = next((model for model in candidates if model.preset == preset), None)
                if standard:
                    return standard.id
        choices = specialized or [
            model
            for model in candidates
            if model.provider in VISION_PROVIDERS
            and model.preset not in {spec.key for spec in HOSTED_PRESETS} | {"nvidia-sol"}
        ]
        if len(choices) == 1:
            return choices[0].id
        if choices:
            raise DomainError(
                "Several models support this annotation: "
                + ", ".join(model.name for model in choices)
                + ". Name the model in your message or select it in the viewer."
            )
        if project.annotation_model_id or available:
            raise DomainError(
                "No configured model can automatically annotate these targets on this image. "
                "Connect a compatible annotation model, or name a model explicitly."
            )
        return None

    def update(self, project_id: str, identifier: str, request: ModelUpdate) -> ModelRecord:
        with self.store.transaction() as session:
            model = session.get(ModelRecord, identifier)
            self.check_editable(model, project_id, request.base_version)
            name = request.name.strip()
            if not name:
                raise DomainError("Give this model a name to use in chat.")
            if any(
                m.id != identifier
                and not m.archived
                and m.name.strip().casefold() == name.casefold()
                for m in session.list(ModelRecord, project_id)
            ):
                raise DomainError("Another model has this name. Choose a different name.")
            updated = model.model_copy(update={"name": name, "version": model.version + 1})
            session.update(updated)
        return updated

    @staticmethod
    def check_editable(model: ModelRecord, project_id: str, version: int) -> None:
        if model.project_id != project_id or model.archived:
            raise DomainError("Model is not available in this project.")
        if model.preset:
            raise DomainError("Preloaded base models cannot be renamed or deleted.")
        if model.version != version:
            raise Conflict("Model changed. Refresh and try again.")

    def delete(self, project_id: str, identifier: str, request: DeleteModelRequest) -> ModelRecord:
        with self.store.transaction() as session:
            model = session.get(ModelRecord, identifier)
            self.check_editable(model, project_id, request.base_version)
            if request.confirmation_name != model.name:
                raise DomainError("Type the model's exact name to confirm deletion.")
            Deletion.idle(session, project_id)
            if request.scope == "model" and model.learner_id:
                Deletion.model_family(
                    session, project_id, model.learner_id, request.related_versions
                )
                return session.get(ModelRecord, identifier)
            project = session.get(Project, project_id)
            if identifier == project.annotation_model_id or identifier in project.defaults.values():
                raise DomainError(
                    "Choose a different annotation default before deleting this model."
                )
            if any(
                learner.initial_model_id == identifier and not learner.archived
                for learner in session.list(Learner, project_id)
            ):
                raise DomainError("Delete the training setup using this starting model first.")
            updated = model.model_copy(update={"archived": True, "version": model.version + 1})
            session.update(updated)
        return updated

    def set_default(self, project_id: str, identifier: str, base_version: int) -> Project:
        model = self.get(project_id, identifier)
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            if project.version != base_version:
                raise Conflict("Project settings changed. Refresh before choosing a default.")
            supported = self.supported_labels(project, model)
            updated = project.model_copy(
                update={
                    "annotation_model_id": model.id,
                    "defaults": {label: model.id for label in supported},
                    "version": project.version + 1,
                }
            )
            session.update(updated)
            return updated

    @classmethod
    def supported_labels(cls, project: Project, model: ModelRecord) -> set[int]:
        """Resolve project targets, including a model's inherited vocabulary."""
        return {
            label.id
            for label in project.labels
            if label.id
            and (
                label.name.casefold() in vista_targets()
                if model.provider == "vista3d" and (model.read_only or model.inherit_targets)
                else cls.promptable(model) or label.id in model.label_ids
            )
        }

    @staticmethod
    def promptable(model: ModelRecord) -> bool:
        return model.provider in {*VISION_PROVIDERS, *SAM_MODELS} or (
            model.provider == "vista3d" and (model.read_only or model.inherit_targets)
        )

    @classmethod
    def validate_targets(cls, model: ModelRecord, names: list[str]) -> None:
        if not cls.promptable(model):
            raise DomainError(
                "This model has fixed targets. Select a promptable model "
                "to annotate a new structure."
            )
        if model.provider == "vista3d":
            missing = [name for name in names if name.casefold() not in vista_targets()]
            if missing:
                raise DomainError(
                    "VISTA3D automatic CT segmentation does not support: " + ", ".join(missing)
                )

    @classmethod
    def for_labels(cls, model: ModelRecord, label_ids: list[int]) -> ModelRecord:
        if cls.promptable(model) or model.provider == "vista3d":
            return model.model_copy(update={"label_ids": [0] + label_ids})
        return model

    @staticmethod
    def requires_2d(model: ModelRecord) -> bool:
        return model.provider in {"huggingface", *VISION_PROVIDERS} or (
            model.provider == "monai-unet" and model.config.get("spatial_dims", 3) == 2
        )

    @staticmethod
    def requires_3d(model: ModelRecord) -> bool:
        return model.provider == "vista3d" or (
            model.provider == "monai-unet" and model.config.get("spatial_dims", 3) == 3
        )

    @staticmethod
    def requires_spatial(model: ModelRecord) -> bool:
        return model.provider in SAM_MODELS

    @staticmethod
    def spatial_provider() -> PromptedSegmenter:
        try:
            from monailabel.sam.runtime import SamSegmenter
        except ImportError as exc:
            raise DomainError(
                "The local SAM runtime could not be loaded. Start from the repository "
                "with uv run monailabel-server to install the required dependencies."
            ) from exc
        return SamSegmenter()

    def register(self, project_id: str, request: ModelRegister) -> ModelRecord:
        project = self.store.get(Project, project_id)
        ids = request.label_ids
        if len(ids) != len(set(ids)) or 0 not in ids:
            raise DomainError("Model labels must be unique and include background 0.")
        if not set(ids) <= {label.id for label in project.labels}:
            raise DomainError("Model label IDs must exist in this project.")
        name = request.name.strip()
        if not name:
            raise DomainError("Give this model a name to use in chat.")
        model = ModelRecord(
            project_id=project_id, **request.model_copy(update={"name": name}).model_dump()
        )
        if not self.promptable(model) and len(ids) < 2:
            raise DomainError("Fixed-label models need their supported foreground labels.")
        config = parse_config(model)
        if config.credential_id:
            self.credentials(project_id, config.credential_id)
        with self.store.transaction() as session:
            if any(
                not m.archived and m.name.strip().casefold() == name.casefold()
                for m in session.list(ModelRecord, project_id)
            ):
                raise DomainError("Another model has this name. Choose a different name.")
            session.insert(model)
        return model

    def predict(
        self,
        project: Project,
        model: ModelRecord,
        image: Image,
        prompt: str = "",
        affine: list[list[float]] | None = None,
    ) -> Mask:
        if self.requires_spatial(model):
            raise DomainError(
                "SAM requires viewer spatial prompts; "
                "automatic unprompted evaluation is unavailable."
            )
        if model.provider in {recipe.id for recipe in self.recipes.list()}:
            if model.state_key is None and not (model.provider == "vista3d" and model.read_only):
                raise DomainError("Model has no trained state.")
            provider: Segmenter | VolumeSegmenter = self.recipes.segmenter(
                model.provider, self.artifacts.json(model.state_key) if model.state_key else {}
            )
        else:
            if model.provider not in self.providers:
                raise DomainError(f"Unknown provider '{model.provider}'.")
            provider = self.providers[model.provider]
        labels = [label for label in project.labels if label.id in model.label_ids]
        if isinstance(provider, VolumeSegmenter):
            if affine is None:
                raise DomainError("This model requires the original volume geometry.")
            result = provider.predict_volume(Volume(image, affine), labels, prompt, model).mask
        else:
            result = provider.predict(image, labels, prompt, model).mask
        if result.shape != image.shape[:-1] or not set(np.unique(result)) <= set(model.label_ids):
            raise DomainError("Provider returned invalid geometry or label IDs.")
        return result
