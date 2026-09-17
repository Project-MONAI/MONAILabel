"""Native segmentation scores against fixed, reviewed, held-out references."""

from collections.abc import Callable

import numpy as np

from monailabel.core.errors import DomainError
from monailabel.core.evaluation import EvaluationSetVersion
from monailabel.core.models import (
    Asset,
    ModelMetrics,
    ModelRecord,
    Project,
    Sample,
    Snapshot,
    Split,
)
from monailabel.server.lineage import training_identity
from monailabel.server.models import Models
from monailabel.server.storage import Artifacts, Store


class ValidationScorer:
    def __init__(self, store: Store, artifacts: Artifacts, models: Models):
        self.store, self.artifacts, self.models = store, artifacts, models

    def samples(
        self,
        project: Project,
        model: ModelRecord,
        snapshot: Snapshot | EvaluationSetVersion,
        label_ids: list[int],
    ) -> list[Sample]:
        if (
            snapshot.project_id != project.id
            or snapshot.protocol_version != project.protocol_version
        ):
            raise DomainError("Evaluation references must match this project's protocol.")
        if not label_ids or not set(label_ids) <= {
            label.id for label in snapshot.labels if label.id
        }:
            raise DomainError("Evaluation references must cover the model's foreground structures.")
        validation = [s for s in snapshot.samples if s.split == Split.VALIDATION]
        if not validation:
            raise DomainError("No reviewed evaluation cases were saved for this run.")
        if any(s.label_source != "reviewed" or not s.decision_id for s in validation):
            raise DomainError("Evaluation requires accepted reference annotations.")
        used_groups, used_images = training_identity(self.store, model)
        if any(s.group_id in used_groups or s.image_key in used_images for s in validation):
            raise DomainError("Evaluation source groups overlap model training lineage.")
        return validation

    def score(
        self,
        project: Project,
        model: ModelRecord,
        validation: list[Sample],
        label_ids: list[int],
        progress: Callable[[int, int], None],
    ) -> tuple[ModelMetrics, dict[int, float]]:
        model = self.models.for_labels(model, label_ids)
        intersections = {label: 0 for label in label_ids}
        denominators = {label: 0 for label in label_ids}
        references = {label: 0 for label in label_ids}
        for index, sample in enumerate(validation):
            progress(index, len(validation))
            predicted = self.models.predict(
                project,
                model,
                self.artifacts.array(sample.image_key),
                project.instructions,
                sample.affine or self.store.get(Asset, sample.asset_id).affine,
            )
            reference = self.artifacts.array(sample.mask_key)
            for label in label_ids:
                truth, output = reference == label, predicted == label
                intersections[label] += int(np.count_nonzero(truth & output))
                denominators[label] += int(truth.sum() + output.sum())
                references[label] += int(truth.sum())
        if any(count == 0 for count in references.values()):
            raise DomainError(
                "Evaluation must contain reviewed examples of every target structure."
            )
        dice = {label: 2 * intersections[label] / denominators[label] for label in label_ids}
        iou = {
            label: intersections[label] / (denominators[label] - intersections[label])
            for label in label_ids
        }
        return ModelMetrics(per_class=dice, mean_dice=sum(dice.values()) / len(dice)), iou
