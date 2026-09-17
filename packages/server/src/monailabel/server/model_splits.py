"""Model-owned, growing train/validation membership over shared project data."""

import hashlib
import math

from monailabel.core.errors import Conflict, DomainError, NotFound
from monailabel.core.evaluation import ModelSplit
from monailabel.core.models import (
    Annotation,
    Asset,
    Job,
    Learner,
    ModelRecord,
    Project,
    Proposal,
    ReviewDecision,
    Snapshot,
    Split,
)
from monailabel.server.evaluation_sets import components, reserved
from monailabel.server.storage import Session


def assign(
    session: Session,
    project: Project,
    learner_id: str,
    percentage: int,
    required: set[int],
    allow_predictions: bool,
    parent_id: str | None,
    *,
    external_validation: bool = False,
) -> tuple[ModelSplit, dict[str, Split]]:
    learner = session.get(Learner, learner_id)
    if learner.project_id != project.id or learner.archived:
        raise DomainError("Choose an active model from this project.")
    if any(
        j.kind == "train"
        and j.status in {"queued", "running"}
        and j.request.get("learner_id") == learner_id
        for j in session.list(Job, project.id)
    ):
        raise Conflict("This model is already training. New cases will join its next run.")
    try:
        record = session.get(ModelSplit, learner_id)
    except NotFound:
        record = ModelSplit(id=learner_id, project_id=project.id, learner_id=learner_id)
    assets = session.list(Asset, project.id)
    decisions = {d.annotation_id: d for d in session.list(ReviewDecision, project.id)}
    eligible: set[str] = set()
    accepted: set[str] = set()
    for asset in assets:
        if not asset.annotation_id:
            continue
        annotation = session.get(Annotation, asset.annotation_id)
        if not required <= set(annotation.covered_labels):
            continue
        decision = decisions.get(annotation.id)
        if decision and decision.verdict == "accepted":
            accepted.add(asset.id)
            eligible.add(asset.id)
        elif allow_predictions and decision is None and annotation.proposal_id:
            proposal = session.get(Proposal, annotation.proposal_id)
            if (
                proposal.slice is None
                and proposal.image_region is None
                and proposal.mask_key == annotation.mask_key
                and required - {0} <= set(proposal.label_ids)
            ):
                eligible.add(asset.id)
    excluded_groups, excluded_images = reserved(session, project.id)
    for asset in assets:
        if asset.split == Split.VALIDATION:
            excluded_groups.add(asset.group_id)
            excluded_images.add(asset.image_key)
    grouped = {
        key: items
        for key, items in components(assets).items()
        if any(a.id in eligible for a in items)
        and not any(a.group_id in excluded_groups or a.image_key in excluded_images for a in items)
    }
    if not grouped and external_validation:
        raise DomainError("Accept a complete training annotation outside the evaluation set first.")
    if len(grouped) < 2 and not external_validation:
        raise DomainError(
            "This model needs at least two independent cases with complete labels. "
            "Review labels or import more images for annotation. "
            "Evaluation-only cases stay separate."
        )
    training = set(record.training_groups)
    validation = set(record.validation_groups)
    training_images = set(record.training_image_keys)
    validation_images = set(record.validation_image_keys)
    # Model lineage and past jobs remain training even if their assets were removed.
    models = {m.id: m for m in session.list(ModelRecord, project.id)}
    ancestors: set[str] = set()
    while parent_id and parent_id in models and parent_id not in ancestors:
        ancestors.add(parent_id)
        parent_id = models[parent_id].parent_id
    for model in models.values():
        if model.learner_id == learner_id or model.id in ancestors:
            training.update(model.training_groups)
            if model.snapshot_id:
                snapshot = session.get(Snapshot, model.snapshot_id)
                training_images.update(
                    s.image_key for s in snapshot.samples if s.split == Split.TRAIN
                )
    for job in session.list(Job, project.id):
        if job.kind == "train" and job.request.get("learner_id") == learner_id:
            snapshot_id = job.request.get("snapshot_id")
            if isinstance(snapshot_id, str):
                for sample in session.get(Snapshot, snapshot_id).samples:
                    if sample.split == Split.TRAIN:
                        training.add(sample.group_id)
                        training_images.add(sample.image_key)
    selected: set[str] = set()
    fixed_training: set[str] = set()
    for key, items in grouped.items():
        in_validation = any(
            a.group_id in validation or a.image_key in validation_images for a in items
        )
        in_training = any(a.group_id in training or a.image_key in training_images for a in items)
        if in_validation and in_training:
            raise Conflict(
                "A patient or duplicate image joins this model's training and validation lists. "
                "Keep related cases together; use a new model split for this lineage."
            )
        if in_validation:
            selected.add(key)
        elif in_training:
            fixed_training.add(key)
    candidates = [
        key
        for key, items in grouped.items()
        if key not in selected | fixed_training and any(a.id in accepted for a in items)
    ]
    candidates.sort(key=lambda key: hashlib.sha256(f"{learner_id}:{key}".encode()).digest())
    target = min(len(grouped) - 1, max(1, math.ceil(len(grouped) * percentage / 100)))
    if not external_validation:
        selected.update(candidates[: max(0, target - len(selected))])
    assignments: dict[str, Split] = {}
    for key, items in grouped.items():
        held_out = key in selected
        for asset in items:
            (validation if held_out else training).add(asset.group_id)
            (validation_images if held_out else training_images).add(asset.image_key)
            if asset.id in (accepted if held_out else eligible):
                assignments[asset.id] = Split.VALIDATION if held_out else Split.TRAIN
    if not external_validation and Split.VALIDATION not in assignments.values():
        raise DomainError(
            "No unused, accepted cases are available for this model's validation. "
            "Import and review new cases; previously trained cases cannot become validation."
        )
    if Split.TRAIN not in assignments.values():
        raise DomainError("Add accepted training cases before starting this model.")
    updates = {
        "validation_mode": "fixed" if external_validation else "percentage",
        "label_ids": sorted(required),
        "validation_percentage": percentage,
        "training_groups": sorted(training),
        "validation_groups": sorted(validation),
        "training_image_keys": sorted(training_images),
        "validation_image_keys": sorted(validation_images),
    }
    if any(getattr(record, key) != value for key, value in updates.items()):
        record = record.model_copy(update=updates | {"version": record.version + 1})
    return record, assignments


def refresh_percentage_sets(session: Session, project_id: str) -> None:
    """Extend each model's saved percentage set when new labels are accepted."""
    project = session.get(Project, project_id)
    for record in session.list(ModelSplit, project_id):
        learner = session.get(Learner, record.learner_id)
        required = set(record.label_ids or learner.label_ids)
        if (
            learner.archived
            or learner.evaluation_set_id
            or record.validation_mode != "percentage"
            or len(required) < 2
        ):
            continue
        try:
            updated, _ = assign(
                session,
                project,
                learner.id,
                record.validation_percentage,
                required,
                False,
                None,
            )
        except DomainError:
            # Reviewing a case must not depend on training readiness. Active runs,
            # incomplete cohorts or conflicting aliases are checked on the next run.
            continue
        if updated != record:
            session.update(updated)
