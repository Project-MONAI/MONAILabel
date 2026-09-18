"""Atomic image/label imports with protected evaluation membership."""

import io

import numpy as np
from PIL import Image

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.evaluation import EvaluationReservation, EvaluationSet
from monailabel.core.models import Annotation, Asset, Project, ReviewDecision, Split
from monailabel.core.reference_imports import (
    EvaluationImportResult,
    LabelImportRequest,
    ReferenceImport,
)
from monailabel.core.video import VideoAsset
from monailabel.server.data import MAX_FILE_BYTES, MAX_IMAGE_PIXELS, decode_image
from monailabel.server.evaluation_sets import EvaluationSets, components, reserved, training_history
from monailabel.server.labels import imported_labels
from monailabel.server.model_splits import refresh_percentage_sets
from monailabel.server.storage import Artifacts, Store


def decode_labels(name: str, content: bytes) -> tuple[np.ndarray, list[list[float]] | None]:
    if len(content) > MAX_FILE_BYTES:
        raise DomainError("Label file exceeds 128 MiB.", status=413)
    if name.lower().endswith((".nii", ".nii.gz")):
        values, affine = decode_image(name, content)
        values = values[..., 0]
    else:
        if not name.lower().endswith((".png", ".tif", ".tiff")):
            raise DomainError("Reference labels must be NIfTI, PNG or TIFF label maps.")
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise DomainError("Label image exceeds the local pixel limit.")
                values = np.asarray(image).copy()
        except Image.DecompressionBombError as exc:
            raise DomainError("Label image exceeds the local pixel limit.", status=413) from exc
        except (OSError, ValueError) as exc:
            raise DomainError("Could not read the reference label file.") from exc
        affine = None
    if (
        values.ndim not in {2, 3}
        or not np.isfinite(values).all()
        or np.any(values != np.rint(values))
    ):
        raise DomainError("Reference labels must contain integer structure values.")
    if np.any(values < 0) or np.any(values > 65535):
        raise DomainError("Reference label values must be between 0 and 65535.")
    return values.astype(np.uint16), affine


class ReferenceImports:
    def __init__(self, store: Store, artifacts: Artifacts):
        self.store, self.artifacts = store, artifacts

    def import_pair(
        self,
        project_id: str,
        request: LabelImportRequest,
        image_name: str,
        image_content: bytes,
        label_name: str,
        label_content: bytes,
        user_id: str,
    ) -> EvaluationImportResult:
        if not image_name or not label_name or max(len(image_name), len(label_name)) > 200:
            raise DomainError("Choose image and label files with names up to 200 characters.")
        image, affine = decode_image(image_name, image_content)
        values, geometry = decode_labels(label_name, label_content)
        if values.shape != image.shape[:-1]:
            raise DomainError("Reference label dimensions do not match the image.")
        if (affine is None) != (geometry is None) or (
            affine is not None
            and geometry is not None
            and not np.allclose(affine, geometry, rtol=1e-5, atol=1e-4)
        ):
            raise DomainError("Reference label orientation/spacing does not match the image.")
        if not set(np.unique(values)) <= {0, *request.labels}:
            raise DomainError("Give a structure name for every foreground value in the label file.")
        image_key = self.artifacts.put_array(image)
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            project, mapping = imported_labels(project, request.labels)
            record: EvaluationSet | None = None
            is_new_set = False
            if request.evaluation_set_id:
                record = EvaluationSets.get(session, project_id, request.evaluation_set_id)
                if record.version != request.base_version:
                    raise Conflict("The evaluation set changed. Refresh before importing.")
                if record.archived:
                    raise Conflict("Restore the evaluation set before importing.")
                is_new_set = False
            elif request.split == Split.VALIDATION:
                record = EvaluationSet(
                    project_id=project_id,
                    name=EvaluationSets.check_name(
                        session, project_id, request.evaluation_set_name or ""
                    ),
                    percentage=100,
                    auto_update=False,
                )
                is_new_set = True
            assets = session.list(Asset, project_id)
            same_image = [a for a in assets if a.image_key == image_key]
            known = {a.group_id for a in same_image}
            group = request.group_id or (
                next(iter(known)) if len(known) == 1 else f"image:{image_key}"
            )
            if len(known) > 1 and request.group_id is None:
                raise Conflict("This image has multiple patient IDs. Specify which ID to use.")
            asset = next((a for a in same_image if a.group_id == group), None)
            new_asset = asset is None
            if asset is None:
                asset = Asset(
                    project_id=project_id,
                    name=image_name,
                    group_id=group,
                    split=request.split,
                    image_key=image_key,
                    spatial_shape=list(image.shape[:-1]),
                    kind="volume3d" if affine else "image2d",
                    affine=affine,
                    source_key=self.artifacts.put(image_content),
                )
            linked = next(
                group_assets
                for group_assets in components([*assets, *([asset] if new_asset else [])]).values()
                if any(a.id == asset.id for a in group_assets)
            )
            videos = [v for v in session.list(VideoAsset, project_id) if v.group_id == group]
            if any(
                v.split == Split.TRAIN if record is not None else v.split != request.split
                for v in videos
            ):
                raise Conflict("Related video clips already have a different dataset use.")
            if record is not None:
                used_groups, used_images = training_history(session, project_id)
                if any(a.group_id in used_groups or a.image_key in used_images for a in linked):
                    raise Conflict(
                        "This patient or image was already used for training "
                        "and cannot be an evaluation reference."
                    )
                if any(a.split == Split.TRAIN for a in linked):
                    raise Conflict(
                        "This patient is assigned to training "
                        "and cannot be imported for evaluation."
                    )
            else:
                groups, images = reserved(session, project_id)
                if any(
                    a.group_id in groups or a.image_key in images or a.split == Split.VALIDATION
                    for a in linked
                ):
                    raise Conflict("This image is reserved for evaluation. Choose Evaluation.")
                if any(a.split != request.split for a in linked):
                    raise Conflict(
                        "This patient already has a different use in the project. "
                        "Choose the same use or update the existing dataset first."
                    )
            mask = np.zeros(values.shape, dtype=np.uint8)
            for source, target in mapping.items():
                mask[values == source] = target
            mask_key = self.artifacts.put_array(mask)
            covered = [0, *mapping.values()]
            annotation = (
                session.get(Annotation, asset.annotation_id) if asset.annotation_id else None
            )
            if annotation and (
                annotation.mask_key != mask_key or set(annotation.covered_labels) != set(covered)
            ):
                raise Conflict(
                    "This image already has different annotations. Existing annotations were kept."
                )
            if annotation is None:
                annotation = Annotation(
                    project_id=project_id,
                    asset_id=asset.id,
                    revision=asset.revision + 1,
                    mask_key=mask_key,
                    covered_labels=covered,
                    reviewer=user_id,
                )
                session.insert(annotation)
                asset = asset.model_copy(
                    update={"revision": annotation.revision, "annotation_id": annotation.id}
                )
            if new_asset:
                session.insert(asset)
            else:
                session.update(asset)
            if record is not None:
                # Reserve every linked patient and decoded-image duplicate, without sampling.
                reservations = {
                    r.group_id: r for r in session.list(EvaluationReservation, project_id)
                }
                groups = {a.group_id for a in linked}
                keys = {a.image_key for a in linked}
                for group_id in groups:
                    old = reservations.get(group_id)
                    reservation = EvaluationReservation(
                        project_id=project_id, group_id=group_id, image_keys=sorted(keys)
                    )
                    if old:
                        session.update(
                            old.model_copy(
                                update={"image_keys": sorted(set(old.image_keys) | keys)}
                            )
                        )
                    else:
                        session.insert(reservation)
                for item in linked:
                    current = asset if item.id == asset.id else item
                    if current.split != Split.VALIDATION:
                        session.update(current.model_copy(update={"split": Split.VALIDATION}))
                for video in videos:
                    if video.split != Split.VALIDATION:
                        session.update(video.model_copy(update={"split": Split.VALIDATION}))
                members = sorted(set(record.member_groups) | groups)
                cohort = sorted(set(record.cohort_groups) | groups)
                if members != record.member_groups or cohort != record.cohort_groups:
                    record = record.model_copy(
                        update={
                            "member_groups": members,
                            "cohort_groups": cohort,
                            "version": record.version + 1,
                        }
                    )
                if is_new_set:
                    session.insert(record)
                else:
                    session.update(record)
            else:
                EvaluationSets.on_import(session, asset, is_new=new_asset, apply_policies=False)
            session.update(project)
            decisions = [
                d
                for d in session.list(ReviewDecision, project_id)
                if d.annotation_id == annotation.id
            ]
            if request.reviewed and (not decisions or decisions[-1].verdict != "accepted"):
                session.insert(
                    ReviewDecision(
                        project_id=project_id,
                        asset_id=asset.id,
                        annotation_id=annotation.id,
                        revision=annotation.revision,
                        reviewer_id=user_id,
                        verdict="accepted",
                        comment="External reference labels confirmed as reviewed during import.",
                    )
                )
                refresh_percentage_sets(session, project_id)
            session.insert(
                ReferenceImport(
                    project_id=project_id,
                    asset_id=asset.id,
                    annotation_id=annotation.id,
                    evaluation_set_id=record.id if record else None,
                    image_name=image_name,
                    label_name=label_name,
                    source_label_key=self.artifacts.put(label_content),
                    label_mapping=mapping,
                    reviewed=request.reviewed,
                    imported_by=user_id,
                    source=request.source,
                )
            )
            return EvaluationImportResult(
                asset_id=asset.id,
                annotation_id=annotation.id,
                covered_labels=covered,
                evaluation_set=record,
            )
