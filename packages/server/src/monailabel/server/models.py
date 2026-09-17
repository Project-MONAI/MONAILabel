"""Provider registry and validated execution; explicit model choices never fall back."""

from collections.abc import Callable

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
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
from monailabel.providers.classification import RemoteClassifier
from monailabel.providers.local import ThresholdSegmenter
from monailabel.providers.remote import RemoteSegmenter, parse_config
from monailabel.providers.sam import MODELS as SAM_MODELS
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
            "openai-polygons": RemoteSegmenter("openai-polygons", credentials),
            "openai-chat-polygons": RemoteSegmenter("openai-chat-polygons", credentials),
        }
        self.classifiers: dict[str, Classifier] = {
            name: RemoteClassifier(credentials)
            for name in ("openai-polygons", "openai-chat-polygons")
        }

    def get(self, project_id: str, identifier: str) -> ModelRecord:
        model = self.store.get(ModelRecord, identifier)
        if model.project_id != project_id:
            raise DomainError("Model belongs to another project.", status=403)
        if model.archived:
            raise DomainError("This model has been deleted from the active catalog.")
        return model

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
        return model.provider in {"openai-polygons", "openai-chat-polygons", *SAM_MODELS} or (
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
        return model.provider in {"huggingface", "openai-polygons", "openai-chat-polygons"} or (
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
        model = ModelRecord(project_id=project_id, **request.model_dump())
        if not self.promptable(model) and len(ids) < 2:
            raise DomainError("Fixed-label models need their supported foreground labels.")
        config = parse_config(model)
        if config.credential_id:
            self.credentials(project_id, config.credential_id)
        with self.store.transaction() as session:
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
