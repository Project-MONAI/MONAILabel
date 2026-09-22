"""Proposal generation and revision-checked human review."""

from collections.abc import Callable

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.geometry import orient_plane, region_pixels, region_slice, restore_plane
from monailabel.core.models import (
    AnnotateRequest,
    Annotation,
    Asset,
    DecisionRequest,
    Job,
    Project,
    Proposal,
    RestoreRequest,
    ReviewDecision,
    ReviewRequest,
)
from monailabel.core.ports import Mask
from monailabel.core.tiling import image_tiles
from monailabel.server import spatial_annotation
from monailabel.server.data import validate_mask
from monailabel.server.image_annotation import annotate_tiles
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.model_splits import refresh_percentage_sets
from monailabel.server.models import Models
from monailabel.server.review_units import ReviewUnits
from monailabel.server.review_units.decisions import decide_source_units
from monailabel.server.storage import Artifacts, Store


class Annotations:
    def __init__(self, store: Store, artifacts: Artifacts, models: Models, jobs: Jobs):
        self.store, self.artifacts, self.models, self.jobs = store, artifacts, models, jobs

    def annotate(self, asset_id: str, request: AnnotateRequest, key: str | None = None) -> Job:
        work = self.prepare(asset_id, request)
        asset = self.store.get(Asset, asset_id)
        payload = request.model_dump(mode="json") | {"asset_id": asset_id}
        return self.jobs.submit("annotate", asset.project_id, payload, work, key)

    def prepare(self, asset_id: str, request: AnnotateRequest) -> Callable[[JobContext], Outcome]:
        """Validate once and capture the source revision for single or batch execution."""
        asset = self.store.get(Asset, asset_id)
        project = self.store.get(Project, asset.project_id)
        scope = request.slice
        crop = request.image_region
        if request.image_tiling and asset.kind != "image2d":
            raise DomainError("Image tiling requires a 2D image.")
        if crop and (
            asset.kind != "image2d"
            or crop.x + crop.width > asset.spatial_shape[1]
            or crop.y + crop.height > asset.spatial_shape[0]
        ):
            raise DomainError("Select a region inside the source 2D image.")
        if scope and (asset.kind != "volume3d" or scope.index >= asset.spatial_shape[scope.axis]):
            raise DomainError("Select a valid source-volume slice.")
        if (
            scope
            and scope.window
            and (not np.isfinite(scope.window).all() or scope.window[1] <= scope.window[0])
        ):
            raise DomainError("Slice window must contain finite increasing intensity bounds.")
        label_ids = request.label_ids or [x.id for x in project.labels if x.id]
        if not label_ids:
            raise DomainError("Choose at least one structure to annotate.")
        if len(set(label_ids)) != len(label_ids) or not set(label_ids) <= {
            x.id for x in project.labels if x.id
        }:
            raise DomainError("Select unique foreground label IDs from this project.")
        assignments: dict[str, list[int]] = {}
        tiles = (
            image_tiles(
                asset.spatial_shape[0],
                asset.spatial_shape[1],
                request.image_tiling,
            )
            if request.image_tiling
            else []
        )
        for label in label_ids:
            identifier = (
                request.model_id or project.defaults.get(label) or project.annotation_model_id
            )
            if not identifier:
                raise DomainError(f"Select a model; label {label} has no default.")
            model = self.models.get(project.id, identifier)
            if label not in model.label_ids and not self.models.promptable(model):
                raise DomainError(f"Selected model does not support label {label}.")
            if model.provider == "vista3d" and (model.read_only or model.inherit_targets):
                self.models.validate_targets(
                    model, [next(x.name for x in project.labels if x.id == label)]
                )
            if self.models.requires_3d(model) and asset.kind != "volume3d":
                raise DomainError("This model requires a scalar NIfTI volume.")
            if self.models.requires_2d(model):
                if asset.kind == "volume3d" and scope is None:
                    raise DomainError(
                        "This model requires a slice plane. Use Slicer to annotate this slice "
                        "or all slices with an explicit intensity window."
                    )
                if scope and scope.window is None:
                    raise DomainError(
                        "This 2D provider requires an explicit slice intensity window."
                    )
            assignments.setdefault(identifier, []).append(label)

        spatial = None
        spatial_models = [
            self.models.get(project.id, identifier)
            for identifier in assignments
            if self.models.requires_spatial(self.models.get(project.id, identifier))
        ]
        if spatial_models:
            if len(assignments) != 1:
                raise DomainError("Use one spatial annotation model and target per request.")
            spatial = spatial_annotation.validate(asset, request, spatial_models[0], label_ids)
            # Validate optional runtime before creating a job.
            self.models.spatial_provider()

        def work(context: JobContext) -> Outcome:
            image = self.artifacts.array(asset.image_key)
            mask = np.zeros(asset.spatial_shape, dtype=np.uint8)
            if asset.annotation_id:
                previous = self.store.get(Annotation, asset.annotation_id)
                mask = self.artifacts.array(previous.mask_key).copy()
            indices = (
                range(asset.spatial_shape[scope.axis])
                if request.all_slices and scope
                else [scope.index if scope else 0]
            )
            steps = len(indices) * len(assignments)
            completed = 0
            volume_predictions: dict[str, Mask] = {}
            if spatial:
                model = spatial_models[0]
                mask = spatial_annotation.predict(
                    self.models.spatial_provider(),
                    image,
                    mask,
                    model,
                    request,
                    spatial,
                    label_ids[0],
                    lambda fraction: context.progress(fraction, model.name),
                )
            if tiles:
                annotate_tiles(
                    image,
                    mask,
                    project,
                    assignments,
                    label_ids,
                    request,
                    tiles,
                    self.models,
                    context,
                )
            for plane, source_index in enumerate([] if tiles or spatial else indices):
                region: list[slice | int] = [slice(None)] * len(asset.spatial_shape)
                if scope:
                    region[scope.axis] = source_index
                if crop:
                    region = [
                        slice(crop.y, crop.y + crop.height),
                        slice(crop.x, crop.x + crop.width),
                    ]
                region_key = tuple(region)
                target = mask[region_key]
                footprint = region_pixels(crop) if crop else np.ones((1,) * target.ndim, dtype=bool)
                target[np.isin(target, label_ids) & footprint] = 0
                input_image = image[region_key]
                if crop and crop.runs is not None:
                    input_image = np.where(footprint[..., None], input_image, 0)
                if scope:
                    input_image = orient_plane(input_image, scope.orientation)
                for identifier, selected_labels in assignments.items():
                    model = self.models.for_labels(
                        self.models.get(project.id, identifier), selected_labels
                    )
                    detail = (
                        f"Slice {plane + 1} of {len(indices)} · {model.name}"
                        if scope
                        else model.name
                    )
                    context.progress(completed / steps, detail)
                    model_image = input_image
                    if scope and scope.window and self.models.requires_2d(model):
                        low, high = scope.window
                        model_image = np.clip((input_image - low) / (high - low), 0, 1)
                    prompt = request.prompt
                    if crop:
                        prompt += (
                            "\nThe input is the selected image crop. Return pixel coordinates "
                            "relative to this crop, not the original image. Segment the targets, "
                            "not the selection outline. Pixels outside an irregular selection "
                            "are blank."
                        )
                    if request.all_slices:
                        prompt += (
                            f"\nThis is source slice {source_index}. Segment only the supplied "
                            "image; return no polygons if the requested structure is absent."
                        )
                    try:
                        if self.models.requires_3d(model) and scope:
                            if identifier not in volume_predictions:
                                volume_predictions[identifier] = self.models.predict(
                                    project, model, image, prompt, asset.affine
                                )
                            result = volume_predictions[identifier][region_key]
                        else:
                            result = self.models.predict(
                                project, model, model_image, prompt, asset.affine
                            )
                    except DomainError as exc:
                        raise DomainError(
                            f"{detail}: {exc}", code=exc.code, status=exc.status
                        ) from exc
                    if scope and not self.models.requires_3d(model):
                        result = restore_plane(result, scope.orientation)
                    selected = np.isin(result, selected_labels) & footprint
                    if np.any(selected & (target != 0) & (target != result)):
                        raise Conflict(
                            "Models or preserved annotations overlap. Review scopes separately."
                        )
                    target[selected] = result[selected]
                    completed += 1
                    context.progress(completed / steps, detail)
            proposal = Proposal(
                project_id=project.id,
                asset_id=asset.id,
                base_revision=asset.revision,
                model_ids=list(assignments),
                label_ids=label_ids,
                mask_key=self.artifacts.put_array(mask),
                prompt=request.prompt,
                slice=None if request.all_slices else scope,
                all_slices=request.all_slices,
                volume_plane=scope if request.all_slices else None,
                image_region=crop,
                image_tiling=request.image_tiling,
                spatial_prompt=spatial,
            )
            return Outcome({"proposal_id": proposal.id}, [proposal])

        return work

    def review(
        self,
        asset_id: str,
        request: ReviewRequest,
        imported_mask: Mask | None = None,
        decision: DecisionRequest | None = None,
    ) -> Annotation:
        with self.store.transaction() as session:
            asset = session.get(Asset, asset_id)
            project = session.get(Project, asset.project_id)
            if asset.revision != request.base_revision:
                raise Conflict("The annotation changed. Reload before applying this review.")
            ids = {x.id for x in project.labels}
            if (
                len(set(request.covered_labels)) != len(request.covered_labels)
                or not set(request.covered_labels) <= ids
            ):
                raise DomainError("Review coverage must use unique project label IDs.")
            proposal = None
            if request.proposal_id:
                proposal = session.get(Proposal, request.proposal_id)
                if proposal.asset_id != asset.id or proposal.status != "pending":
                    raise Conflict("Proposal is not pending for this asset.")
                if proposal.base_revision != asset.revision:
                    raise Conflict("Proposal was generated from an older annotation revision.")
            if imported_mask is not None:
                mask = validate_mask(imported_mask, asset, project)
            elif request.mask is not None:
                mask = validate_mask(request.mask, asset, project)
            elif proposal:
                mask = validate_mask(self.artifacts.array(proposal.mask_key), asset, project)
            else:
                raise DomainError("Provide a corrected mask or a pending proposal.")
            if request.regions:
                if asset.kind != "image2d":
                    raise DomainError("Region submission requires a 2D source image.")
                covered = np.zeros(asset.spatial_shape, dtype=bool)
                for region in request.regions:
                    if (
                        region.y + region.height > asset.spatial_shape[0]
                        or region.x + region.width > asset.spatial_shape[1]
                    ):
                        raise DomainError("The submitted region lies outside the source image.")
                    covered[region_slice(region)] |= region_pixels(region)
                saved = (
                    self.artifacts.array(session.get(Annotation, asset.annotation_id).mask_key)
                    if asset.annotation_id
                    else np.zeros(asset.spatial_shape, dtype=np.uint8)
                )
                mask = np.where(covered, mask, saved).astype(np.uint8)
            previous = None
            if decision is not None:
                if decision.verdict != "accepted" or not asset.annotation_id:
                    raise DomainError(
                        "Complete review requires a submitted annotation and a Good decision."
                    )
                if set(request.covered_labels) != ids:
                    raise DomainError(
                        "Good review must cover the complete image and all project labels."
                    )
                previous = session.get(Annotation, asset.annotation_id)
                if previous.regions is not None or request.regions is not None:
                    raise DomainError(
                        "Review each submitted region separately in the review queue."
                    )
            unchanged = (
                previous is not None
                and set(previous.covered_labels) == ids
                and np.array_equal(mask, self.artifacts.array(previous.mask_key))
            )
            if unchanged and previous is not None:
                annotation = previous
            else:
                annotation = Annotation(
                    project_id=project.id,
                    asset_id=asset.id,
                    revision=asset.revision + 1,
                    mask_key=self.artifacts.put_array(mask),
                    covered_labels=request.covered_labels,
                    reviewer=request.reviewer,
                    proposal_id=request.proposal_id,
                    regions=request.regions,
                )
                session.insert(annotation)
                if request.regions:
                    ReviewUnits(self.store, self.artifacts).submit_regions(
                        session, asset, annotation, request.regions
                    )
                else:
                    ReviewUnits(self.store, self.artifacts).reconcile_regions(
                        session, asset, annotation
                    )
                session.update(
                    asset.model_copy(
                        update={
                            "revision": annotation.revision,
                            "annotation_id": annotation.id,
                        }
                    )
                )
                if proposal:
                    session.update(proposal.model_copy(update={"status": "accepted"}))
            if decision is not None:
                session.insert(
                    ReviewDecision(
                        project_id=project.id,
                        asset_id=asset.id,
                        annotation_id=annotation.id,
                        revision=annotation.revision,
                        reviewer_id=request.reviewer,
                        **decision.model_dump(),
                    )
                )
                decide_source_units(session, project.id, asset.id, decision, request.reviewer)
                refresh_percentage_sets(session, project.id)
            return annotation

    def restore(self, asset_id: str, request: RestoreRequest) -> Annotation:
        with self.store.transaction() as session:
            asset = session.get(Asset, asset_id)
            old = session.get(Annotation, request.annotation_id)
            if old.asset_id != asset_id or request.base_revision != asset.revision:
                raise Conflict("Restore requires a revision from this asset and the current base.")
            restored = Annotation(
                project_id=asset.project_id,
                asset_id=asset.id,
                revision=asset.revision + 1,
                mask_key=old.mask_key,
                covered_labels=old.covered_labels,
                reviewer=request.reviewer,
                restored_from=old.id,
                regions=old.regions,
            )
            session.insert(restored)
            ReviewUnits(self.store, self.artifacts).reconcile_regions(session, asset, restored)
            session.update(
                asset.model_copy(
                    update={
                        "revision": restored.revision,
                        "annotation_id": restored.id,
                    }
                )
            )
            return restored

    def reject(self, proposal_id: str) -> Proposal:
        with self.store.transaction() as session:
            proposal = session.get(Proposal, proposal_id)
            if proposal.status != "pending":
                raise Conflict("Only pending proposals can be rejected.")
            proposal = proposal.model_copy(update={"status": "rejected"})
            session.update(proposal)
            return proposal
