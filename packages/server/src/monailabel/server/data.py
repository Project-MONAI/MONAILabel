"""Image ingestion preserves source bytes and explicit NIfTI voxel-to-world geometry."""

import base64
import gzip
import io
from typing import cast

import nibabel as nib
import numpy as np
from PIL import Image as PILImage

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.evaluation import EvaluationSet, ModelSplit
from monailabel.core.models import (
    Annotation,
    Asset,
    ImageImport,
    ImageMetadata,
    Project,
    ProjectCreate,
    ProjectUpdate,
    Proposal,
    ReviewDecision,
    Sample,
    Snapshot,
    SnapshotRequest,
    Split,
)
from monailabel.core.ports import Image, Mask
from monailabel.core.video import VideoAsset
from monailabel.server.evaluation_sets import EvaluationSets, reserved
from monailabel.server.model_splits import assign as assign_model_split
from monailabel.server.storage import Artifacts, Store
from monailabel.server.training_samples import filter_samples

MAX_VOXELS = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_NIFTI_BYTES = 256 * 1024 * 1024


def decode_image(name: str, content: bytes) -> tuple[Image, list[list[float]] | None]:
    try:
        if len(content) > MAX_FILE_BYTES:
            raise DomainError("Image file exceeds 128 MiB.", status=413)
        if name.lower().endswith((".nii", ".nii.gz")):
            if name.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=io.BytesIO(content)) as stream:
                    content = stream.read(MAX_NIFTI_BYTES + 1)
                if len(content) > MAX_NIFTI_BYTES:
                    raise DomainError("Decompressed NIfTI exceeds 256 MiB.", status=413)
            volume = nib.Nifti1Image.from_bytes(content)
            if len(volume.shape) != 3 or any(size < 1 for size in volume.shape):
                raise DomainError(
                    f"Expected a scalar 3D NIfTI; this image has shape {volume.shape}."
                )
            if np.prod(volume.shape) > MAX_VOXELS:
                raise DomainError(
                    f"NIfTI shape {volume.shape} exceeds {MAX_VOXELS:,} voxels.", status=413
                )
            array = np.asarray(volume.dataobj, dtype=np.float32)
            affine = np.asarray(volume.affine, dtype=float)
            if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
                raise DomainError("NIfTI requires an invertible finite affine.")
            result = array[..., None]
            if not np.isfinite(result).all():
                raise DomainError("Image contains NaN or infinite values.")
            return result, cast(list[list[float]], affine.tolist())
        with PILImage.open(io.BytesIO(content)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise DomainError("Image exceeds the local service pixel limit.")
            result = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return result, None
    except PILImage.DecompressionBombError as exc:
        raise DomainError("Image exceeds the local service pixel limit.", status=413) from exc
    except DomainError:
        raise
    except (ValueError, OSError, EOFError, nib.filebasedimages.ImageFileError) as exc:
        raise DomainError("Could not read this image. Import PNG/JPEG or scalar NIfTI-1.") from exc


def nifti_bytes(array: np.ndarray, affine: list[list[float]], source: bytes | None = None) -> bytes:
    header = None
    if source:
        if source.startswith(b"\x1f\x8b"):
            source = gzip.decompress(source)
        header = nib.Nifti1Image.from_bytes(source).header
        header.set_data_dtype(array.dtype)  # type: ignore[no-untyped-call]
        header.set_slope_inter(1, 0)  # type: ignore[no-untyped-call]
    volume = nib.Nifti1Image(array, np.asarray(affine), header=header)  # type: ignore[no-untyped-call]
    return volume.to_bytes()


def validate_mask(mask: object, asset: Asset, project: Project) -> Mask:
    try:
        values = np.asarray(mask)
    except ValueError as exc:
        raise DomainError("Mask must be a rectangular integer array.") from exc
    if values.shape != tuple(asset.spatial_shape) or not np.issubdtype(values.dtype, np.integer):
        raise DomainError(f"Mask must be an integer array of shape {asset.spatial_shape}.")
    if not set(np.unique(values)) <= {label.id for label in project.labels}:
        raise DomainError("Mask contains IDs outside the project's label definitions.")
    return values.astype(np.uint8)


class Datasets:
    def __init__(self, store: Store, artifacts: Artifacts):
        self.store, self.artifacts = store, artifacts

    def create(self, request: ProjectCreate) -> Project:
        project = Project(**request.model_dump())
        with self.store.transaction() as session:
            session.insert(project)
        return project

    def update(self, project_id: str, request: ProjectUpdate) -> Project:
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            if request.base_version != project.version:
                raise Conflict("Project changed. Refresh before saving your settings.")
            updated = project.model_copy(
                update={
                    "name": request.name,
                    "instructions": request.instructions,
                    "version": project.version + 1,
                }
            )
            session.update(updated)
        return updated

    def import_image(self, project_id: str, request: ImageImport) -> Asset:
        self.store.get(Project, project_id)
        try:
            content = base64.b64decode(request.image_base64, validate=True)
        except ValueError as exc:
            raise DomainError("Invalid base64 image.") from exc
        return self.import_content(project_id, request, content)

    def import_content(self, project_id: str, request: ImageMetadata, content: bytes) -> Asset:
        self.store.get(Project, project_id)
        image, affine = decode_image(request.name, content)
        image_key = self.artifacts.put_array(image)
        asset = Asset(
            project_id=project_id,
            name=request.name,
            group_id=request.group_id or f"image:{image_key}",
            split=request.split,
            image_key=image_key,
            spatial_shape=list(image.shape[:-1]),
            kind="volume3d" if affine else "image2d",
            affine=affine,
            source_key=self.artifacts.put(content),
        )
        with self.store.transaction() as session:
            existing = session.list(Asset, project_id)
            if request.group_id is None:
                known_groups = {old.group_id for old in existing if old.image_key == image_key}
                if len(known_groups) > 1:
                    raise Conflict(
                        "This image has multiple existing patient or slide IDs. "
                        "Specify which ID to use."
                    )
                if known_groups:
                    asset = asset.model_copy(update={"group_id": next(iter(known_groups))})
            groups, images = reserved(session, project_id)
            if asset.group_id in groups or asset.image_key in images:
                if request.split == Split.TRAIN:
                    raise Conflict(
                        "This image is reserved for evaluation and cannot enter training."
                    )
                asset = asset.model_copy(update={"split": Split.VALIDATION})
            # Repeat imports into an automatically partitioned cohort retain its split.
            cohort = {
                g for item in session.list(EvaluationSet, project_id) for g in item.cohort_groups
            }
            if request.split == Split.POOL and asset.group_id in cohort:
                previous = next((a for a in existing if a.group_id == asset.group_id), None)
                if previous:
                    asset = asset.model_copy(update={"split": previous.split})
            for old in existing:
                if old.group_id == asset.group_id and old.split != asset.split:
                    raise Conflict(
                        "A source group cannot cross training, validation, or pool splits."
                    )
                if (
                    old.source_key == asset.source_key
                    and old.name == asset.name
                    and old.group_id == asset.group_id
                    and old.split == asset.split
                ):
                    return old
            for video in session.list(VideoAsset, project_id):
                if video.group_id == asset.group_id and video.split != asset.split:
                    raise Conflict("All clips and images from a procedure must share one split.")
            session.insert(asset)
            asset = EvaluationSets.on_import(
                session, asset, is_new=True, apply_policies=not request.shared
            )
        return asset

    def assign_split(self, asset_id: str, split: Split) -> Asset:
        if split not in {Split.TRAIN, Split.VALIDATION}:
            raise DomainError("Choose training or held-out validation.")
        with self.store.transaction() as session:
            asset = session.get(Asset, asset_id)
            groups, images = reserved(session, asset.project_id)
            if split == Split.TRAIN and (asset.group_id in groups or asset.image_key in images):
                raise Conflict("This sample is reserved for evaluation only.")
            if asset.split != Split.POOL:
                raise Conflict(
                    "Only pool cases can be assigned; training and validation stay separate."
                )
            for item in session.list(Asset, asset.project_id):
                if item.group_id == asset.group_id:
                    if item.split != Split.POOL:
                        raise Conflict(
                            "A source group cannot cross training and validation splits."
                        )
                    session.update(item.model_copy(update={"split": split}))
            for video in session.list(VideoAsset, asset.project_id):
                if video.group_id == asset.group_id:
                    session.update(video.model_copy(update={"split": split}))
            return asset.model_copy(update={"split": split})

    def snapshot(
        self,
        project_id: str,
        label_ids: list[int] | None = None,
        *,
        request: SnapshotRequest | None = None,
        authorized_by: str | None = None,
        learner_id: str | None = None,
        validation_percentage: int = 20,
        external_validation: bool = False,
        parent_model_id: str | None = None,
    ) -> Snapshot:
        request = request or SnapshotRequest(label_ids=label_ids)
        if request.allow_unreviewed_training and not authorized_by:
            raise DomainError("A project manager must authorize unreviewed training labels.")
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            required = (
                set(request.label_ids)
                if request.label_ids is not None
                else {label.id for label in project.labels}
            )
            if (
                0 not in required
                or len(required) < 2
                or not required <= {label.id for label in project.labels}
            ):
                raise DomainError(
                    "Choose project foreground labels and background for the snapshot."
                )
            model_split = None
            assignments = None
            if learner_id:
                model_split, assignments = assign_model_split(
                    session,
                    project,
                    learner_id,
                    validation_percentage,
                    required,
                    request.allow_unreviewed_training,
                    parent_model_id,
                    external_validation=external_validation,
                )
            samples = []
            decisions = {d.annotation_id: d for d in session.list(ReviewDecision, project_id)}
            from monailabel.server.learning_data.cases import accepted_whole, cases, unit_samples

            for case in cases(session, project_id):
                asset = case.source
                split = assignments.get(asset.id) if assignments is not None else asset.split
                if split is None or split == Split.POOL or not asset.annotation_id:
                    continue
                if (
                    learner_id
                    and not validation_percentage
                    and not external_validation
                    and split == Split.VALIDATION
                ):
                    continue
                if accepted_whole(session, case, required, decisions) is None:
                    units = unit_samples(session, self.artifacts, case, required, split, decisions)
                    if units:
                        samples.extend(units)
                        continue
                if not isinstance(asset, Asset):
                    continue
                annotation = session.get(Annotation, asset.annotation_id)
                if annotation.regions:
                    continue
                if not required <= set(annotation.covered_labels):
                    continue
                decision = decisions.get(annotation.id)
                proposal = None
                if decision is None or decision.verdict != "accepted":
                    # The explicit exception covers intact, complete model predictions only.
                    # Rejected work, manual drafts, pool data and validation never bypass review.
                    if (
                        not request.allow_unreviewed_training
                        or split != Split.TRAIN
                        or decision is not None
                        or not annotation.proposal_id
                    ):
                        continue
                    proposal = session.get(Proposal, annotation.proposal_id)
                    if (
                        proposal.slice is not None
                        or proposal.image_region is not None
                        or proposal.mask_key != annotation.mask_key
                        or not (required - {0}) <= set(proposal.label_ids)
                    ):
                        continue
                mask_key = annotation.mask_key
                mask = self.artifacts.array(mask_key)
                if not set(np.unique(mask)) <= required:
                    filtered = np.where(np.isin(mask, list(required)), mask, 0).astype(np.uint8)
                    mask_key = self.artifacts.put_array(filtered)
                samples.append(
                    Sample(
                        asset_id=asset.id,
                        image_key=asset.image_key,
                        mask_key=mask_key,
                        revision=annotation.revision,
                        group_id=asset.group_id,
                        split=split,
                        decision_id=decision.id if decision else None,
                        affine=asset.affine,
                        label_source="model_prediction" if proposal else "reviewed",
                        proposal_id=proposal.id if proposal else None,
                        model_ids=proposal.model_ids if proposal else [],
                        annotation_id=annotation.id,
                    )
                )
            if not any(s.split == Split.TRAIN for s in samples):
                raise DomainError(
                    "A reviewer must accept complete training annotations before snapshot creation."
                )
            samples = filter_samples(session, project_id, samples, request.sample_filter)
            if model_split:
                if any(item.id == model_split.id for item in session.list(ModelSplit, project_id)):
                    session.update(model_split)
                else:
                    session.insert(model_split)
            snapshot = Snapshot(
                evaluation_requested=not learner_id
                or external_validation
                or validation_percentage > 0,
                sample_filter=request.sample_filter,
                model_split_id=model_split.id if model_split else None,
                model_split_version=model_split.version if model_split else None,
                project_id=project_id,
                protocol_version=project.protocol_version,
                labels=[label for label in project.labels if label.id in required],
                samples=samples,
                allow_unreviewed_training=request.allow_unreviewed_training,
                authorized_by=authorized_by,
                note=request.note,
            )
            session.insert(snapshot)
        return snapshot
