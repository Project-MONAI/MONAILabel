# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import public datasets into normal, revision-checked annotation/review workflows."""

import gzip
import io
import json
import math
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
from pydantic import JsonValue

from monailabel.core.dataset_templates import DatasetTemplate, DatasetTemplateImport
from monailabel.core.errors import DomainError
from monailabel.core.evaluation import EvaluationSet
from monailabel.core.models import Asset, ImageMetadata, Job, Project, ReviewRequest, Split
from monailabel.core.reference_imports import EvaluationImportRequest
from monailabel.server.annotation import Annotations
from monailabel.server.data import MAX_NIFTI_BYTES, Datasets, decode_image, nifti_bytes
from monailabel.server.dataset_downloads import Archive, Downloads, Source, sources
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.labels import imported_labels
from monailabel.server.raster_datasets import (
    RASTER_FORMATS,
    binary_mask,
    image_names,
    png_mask,
    source_group,
)
from monailabel.server.reference_imports import ReferenceImports
from monailabel.server.storage import Store
from monailabel.server.video.assets import Videos


class DatasetTemplates:
    def __init__(
        self,
        store: Store,
        datasets: Datasets,
        annotations: Annotations,
        jobs: Jobs,
        cache: Path,
        *,
        references: ReferenceImports,
        videos: Videos,
        legacy_cache: Path | None = None,
    ):
        self.store, self.datasets, self.annotations, self.jobs = store, datasets, annotations, jobs
        self.downloads = Downloads(cache, legacy_cache)
        self.references = references
        self.videos = videos

    def catalog(self) -> list[DatasetTemplate]:
        return [
            DatasetTemplate.model_validate(
                source.model_dump(include=set(DatasetTemplate.model_fields))
                | {
                    "cached": self.downloads.cached(source)
                    and (
                        source.mask_download is None or self.downloads.cached(source.mask_download)
                    )
                }
            )
            for source in sources()
        ]

    def start(self, project_id: str, request: DatasetTemplateImport, user_id: str) -> Job:
        self.store.get(Project, project_id)
        source = next((s for s in sources() if s.id == request.template_id), None)
        if source is None or not source.importable:
            raise DomainError("Choose an importable dataset template.")
        if source.format == "video":
            if source.video is None:
                raise DomainError("This video template is missing its clip metadata.")
            if (
                request.split != Split.POOL
                or request.include_masks
                or request.evaluation_percentage is not None
                or request.evaluation_set_id
                or request.section != "training"
                or request.offset
                or request.channel
                or request.targets
            ):
                raise DomainError(
                    "This sample imports one video for CVAT annotation, without reference tracks. "
                    "Training, evaluation, image channels and case offsets are not available."
                )
        if request.section not in source.sections:
            raise DomainError("This dataset does not have the requested source section.")
        needs_masks = request.include_masks or request.evaluation_percentage is not None
        if source.format == "kvasir-instrument" and request.evaluation_percentage is not None:
            raise DomainError(
                "Kvasir-Instrument does not publish patient/procedure IDs. "
                "Import it for annotation/training or as a separate evaluation-only collection; "
                "a frame-based percentage split would not establish independent cases."
            )
        if needs_masks and (not source.has_masks or request.section == "test"):
            raise DomainError(
                "This selection has no reference labels. "
                "Choose a labeled dataset for evaluation, or import images for annotation."
            )
        if request.evaluation_set_id:
            record = self.store.get(EvaluationSet, request.evaluation_set_id)
            if record.project_id != project_id or record.archived:
                raise DomainError("Choose an active evaluation set in this project.")
        if request.channel >= max(1, len(source.channels)):
            raise DomainError("Choose a channel provided by this dataset.")
        if source.format in {"totalsegmentator", "totalsegmentator-mr"} and needs_masks:
            if (
                not request.targets
                or len(set(request.targets)) != len(request.targets)
                or not set(request.targets) <= set(source.targets)
            ):
                raise DomainError("Choose 1–31 distinct structures from this dataset.")
        elif request.targets:
            raise DomainError("Structure subsets are available for TotalSegmentator masks.")
        if source.format == "image" and request.offset:
            raise DomainError("This sample contains one image; start at case 1.")

        def work(context: JobContext) -> Outcome:
            path = self.downloads.fetch(source, context)
            if source.format == "video":
                assert source.video is not None
                context.progress(0.8, "Reading video frames and importing the CVAT sample")
                video = self.videos.import_file(project_id, source.video, path)
                return Outcome(
                    {
                        "video_ids": [video.id],
                        "asset_ids": [],
                        "annotation_ids": [],
                        "failed": [],
                        "source_url": source.source_url,
                    }
                )
            if source.format == "image":
                context.progress(0.8, "Importing pathology image")
                asset = self.datasets.import_content(
                    project_id,
                    ImageMetadata(
                        name=Path(source.url).name,
                        split=request.split,
                        group_id="openslide:CMU-1",
                        shared=True,
                    ),
                    path.read_bytes(),
                )
                return Outcome(
                    {
                        "asset_ids": [asset.id],
                        "annotation_ids": [],
                        "failed": [],
                        "source_url": source.source_url,
                    }
                )
            return self.import_archive(project_id, request, source, path, user_id, context)

        return self.jobs.submit(
            "dataset_import",
            project_id,
            request.model_dump(mode="json")
            | {
                "source_url": source.source_url,
                "source_checksum": source.checksum,
                "labels_url": source.labels_url,
                "license": source.license,
                "license_url": source.license_url,
                "citation": source.citation,
                "grouping": source.grouping,
                "mask_checksum": source.mask_download.checksum if source.mask_download else None,
                "label_groups": source.label_groups,
            },
            work,
        )

    def import_archive(
        self,
        project_id: str,
        request: DatasetTemplateImport,
        source: Source,
        path: Path,
        user_id: str,
        context: JobContext,
    ) -> Outcome:
        archive = Archive(path)
        mask_archive = None
        needs_masks = request.include_masks or request.evaluation_percentage is not None
        try:
            if needs_masks and source.mask_download:
                mask_archive = Archive(self.downloads.fetch(source.mask_download, context))
            if source.format == "msd":
                section = "imagesTr" if request.section == "training" else "imagesTs"
                images = sorted(
                    name
                    for name in archive.names
                    if PurePosixPath(name).parent.name == section
                    and name.endswith((".nii", ".nii.gz"))
                )
                label_names: dict[int, str] = {}
                if needs_masks:
                    metadata = [
                        name for name in archive.names if PurePosixPath(name).name == "dataset.json"
                    ]
                    if len(metadata) != 1:
                        raise DomainError("Dataset must contain one dataset.json label mapping.")
                    definition = json.loads(archive.read(metadata[0], 2 * 1024**2))
                    label_names = {
                        int(k): str(v) for k, v in definition["labels"].items() if int(k)
                    }
                    if source.label_groups:
                        members = [i for group in source.label_groups.values() for i in group]
                        if len(set(members)) != len(members) or set(members) != set(label_names):
                            raise DomainError(
                                "Template label groups do not match the published dataset."
                            )
                        label_names = {i: name for i, name in enumerate(source.label_groups, 1)}
            elif source.format in RASTER_FORMATS:
                images = image_names(source, archive)
                label_names = {1: "nuclei" if source.format == "tnbc" else "instrument"}
            else:
                images = sorted(
                    name
                    for name in archive.names
                    if PurePosixPath(name).name
                    == ("mri.nii.gz" if source.format == "totalsegmentator-mr" else "ct.nii.gz")
                )
                label_names = (
                    {i + 1: name for i, name in enumerate(request.targets)} if needs_masks else {}
                )
            end = request.offset + request.limit if request.limit is not None else None
            selected = images[request.offset : end]
            if not selected:
                raise DomainError(
                    "No cases match this selection. "
                    "Reduce the starting case or choose another section."
                )
            evaluation_groups: set[str] = set()
            if request.evaluation_percentage is not None:
                groups = list(dict.fromkeys(source_group(source, name) for name in selected))
                if len(groups) < 2:
                    raise DomainError(
                        "A combined import needs at least two independent source groups."
                    )
                count = min(
                    len(groups) - 1,
                    math.ceil(len(groups) * request.evaluation_percentage / 100),
                )
                evaluation_groups = set(groups[:count])
                evaluation_count = sum(
                    source_group(source, name) in evaluation_groups for name in selected
                )
                context.progress(
                    0.5,
                    f"Importing {len(selected) - evaluation_count} cases for annotation/training "
                    f"and {evaluation_count} images with labels for evaluation",
                )
            ids: list[JsonValue] = []
            annotations: list[JsonValue] = []
            annotation_assets: list[JsonValue] = []
            evaluation_assets: list[JsonValue] = []
            failed: list[JsonValue] = []
            evaluation_set = None
            set_name = f"{source.name} evaluation"[:120]
            if request.split == Split.VALIDATION or evaluation_groups:
                evaluation_set = next(
                    (
                        record
                        for record in self.store.list(EvaluationSet, project_id)
                        if not record.archived
                        and (
                            record.id == request.evaluation_set_id
                            if request.evaluation_set_id
                            else record.name == set_name
                        )
                    ),
                    None,
                )
                if request.evaluation_set_id and evaluation_set is None:
                    raise DomainError("The evaluation set is no longer active. Refresh and retry.")
            for index, name in enumerate(selected):
                case_request = request
                if evaluation_groups:
                    is_evaluation = source_group(source, name) in evaluation_groups
                    case_request = request.model_copy(
                        update={
                            "evaluation_percentage": None,
                            "split": Split.VALIDATION if is_evaluation else Split.POOL,
                            "include_masks": is_evaluation or request.include_masks,
                        }
                    )
                context.progress(
                    0.5 + 0.49 * index / len(selected),
                    f"Importing case {index + 1} of {len(selected)}",
                )
                try:
                    asset, annotation_id, updated_set = self.import_case(
                        project_id,
                        case_request,
                        source,
                        archive,
                        name,
                        label_names,
                        user_id,
                        evaluation_set,
                        set_name,
                        mask_archive,
                    )
                    if updated_set is not None:
                        evaluation_set = updated_set
                    ids.append(asset.id)
                    if asset.split == Split.VALIDATION:
                        evaluation_assets.append(asset.id)
                    else:
                        annotation_assets.append(asset.id)
                    if annotation_id:
                        annotations.append(annotation_id)
                except (DomainError, ValueError, OSError) as error:
                    failed.append({"case": name, "error": str(error)})
            return Outcome(
                {
                    "asset_ids": ids,
                    "annotation_ids": annotations,
                    "failed": failed,
                    "source_url": source.source_url,
                    "evaluation_set_id": evaluation_set.id if evaluation_set else None,
                    "annotation_asset_ids": annotation_assets,
                    "evaluation_asset_ids": evaluation_assets,
                    "evaluation_percentage": request.evaluation_percentage,
                }
            )
        finally:
            archive.close()
            if mask_archive:
                mask_archive.close()

    def import_case(
        self,
        project_id: str,
        request: DatasetTemplateImport,
        source: Source,
        archive: Archive,
        name: str,
        label_names: dict[int, str],
        user_id: str,
        evaluation_set: EvaluationSet | None,
        set_name: str,
        mask_archive: Archive | None = None,
    ) -> tuple[Asset, str | None, EvaluationSet | None]:
        content = archive.read(name)
        if source.channels:
            content = select_channel(content, request.channel)
        case = PurePosixPath(name)
        filename = case.name
        if source.format in {"totalsegmentator", "totalsegmentator-mr"}:
            filename = case.parent.name + ".nii.gz"
        elif source.channels:
            filename = (
                filename.removesuffix(".nii.gz").removesuffix(".nii")
                + f"_channel{request.channel}.nii.gz"
            )
        image, affine = decode_image(filename, content)
        mask = np.zeros(image.shape[:-1], dtype=np.uint8) if request.include_masks else None
        if mask is not None:
            if source.format == "msd":
                mask_name = str(case.parent.parent / "labelsTr" / case.name)
                values = read_mask(archive, mask_name, affine, mask.shape)
                original_ids = (
                    {i for group in source.label_groups.values() for i in group}
                    if source.label_groups
                    else set(label_names)
                )
                if not set(np.unique(values)) <= {0, *original_ids}:
                    raise DomainError("Reference mask contains IDs absent from dataset.json.")
                if source.label_groups:
                    for identifier, source_ids in enumerate(source.label_groups.values(), 1):
                        mask[np.isin(values, source_ids)] = identifier
                else:
                    mask = values.astype(np.uint8)
            elif source.format in RASTER_FORMATS:
                mask = binary_mask(source, mask_archive or archive, name, mask.shape)
            else:
                for identifier, label in label_names.items():
                    values = read_mask(
                        archive,
                        str(case.parent / "segmentations" / (label + ".nii.gz")),
                        affine,
                        mask.shape,
                    )
                    if not set(np.unique(values)) <= {0, 1}:
                        raise DomainError("Expected a binary structure mask.")
                    selected = values == 1
                    if np.any(selected & (mask != 0)):
                        raise DomainError(
                            "Selected structure masks overlap; "
                            "choose mutually exclusive structures."
                        )
                    mask[selected] = identifier
        # Match known image identity before assigning a canonical dataset source group.
        key = self.datasets.artifacts.put_array(image)
        group = source_group(source, name)
        known = {a.group_id for a in self.store.list(Asset, project_id) if a.image_key == key}
        if len(known) > 1:
            raise DomainError("This case already has conflicting source groups in the project.")
        if known:
            group = next(iter(known))
        if request.split == Split.VALIDATION:
            assert mask is not None
            result = self.references.import_pair(
                project_id,
                EvaluationImportRequest(
                    evaluation_set_id=evaluation_set.id if evaluation_set else None,
                    base_version=evaluation_set.version if evaluation_set else None,
                    evaluation_set_name=None if evaluation_set else set_name,
                    labels=label_names,
                    group_id=group,
                    source=source.source_url,
                ),
                filename,
                content,
                filename.removesuffix(".gz") if affine is not None else case.stem + "-labels.png",
                nifti_bytes(mask, affine) if affine is not None else png_mask(mask),
                user_id,
            )
            return (
                self.store.get(Asset, result.asset_id),
                result.annotation_id,
                result.evaluation_set,
            )
        mapping = self.ensure_labels(project_id, label_names) if mask is not None else {}
        asset = self.datasets.import_content(
            project_id,
            ImageMetadata(name=filename, group_id=group, split=request.split, shared=True),
            content,
        )
        if mask is None or asset.annotation_id:
            return asset, None, None  # Never overwrite an existing annotation on retry.
        remapped = np.zeros(mask.shape, dtype=np.uint8)
        for original, identifier in mapping.items():
            remapped[mask == original] = identifier
        annotation = self.annotations.review(
            asset.id,
            ReviewRequest(base_revision=0, reviewer=user_id, covered_labels=[0, *mapping.values()]),
            imported_mask=remapped,
        )
        return asset, annotation.id, None

    def ensure_labels(self, project_id: str, names: dict[int, str]) -> dict[int, int]:
        if (
            not names
            or len(names) > 31
            or any(key < 1 or key > 255 or not value.strip() for key, value in names.items())
        ):
            raise DomainError(
                "Dataset labels must define 1–31 foreground structures with IDs 1–255."
            )
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            updated, mapping = imported_labels(project, names)
            if updated != project:
                session.update(updated)
        return mapping


def read_mask(
    archive: Archive, name: str, affine: list[list[float]] | None, shape: tuple[int, ...]
) -> np.ndarray:
    values, geometry = decode_image(name, archive.read(name))
    if (
        geometry is None
        or affine is None
        or values.shape[:-1] != shape
        or not np.allclose(geometry, affine, rtol=1e-5, atol=1e-4)
    ):
        raise DomainError(
            "Reference mask geometry differs from its image. Resample explicitly before importing."
        )
    mask = values[..., 0]
    if not np.array_equal(mask, np.round(mask)):
        raise DomainError("Reference masks require integer label values.")
    return mask


def select_channel(content: bytes, channel: int) -> bytes:
    if content.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as stream:
            content = stream.read(MAX_NIFTI_BYTES + 1)
    if len(content) > MAX_NIFTI_BYTES:
        raise DomainError("Multichannel image exceeds the 256 MiB decompression limit.")
    volume = nib.Nifti1Image.from_bytes(content)
    if len(volume.shape) != 4 or channel >= volume.shape[3]:
        raise DomainError("The selected modality is absent from this image.")
    values = np.asarray(volume.dataobj, dtype=np.float32)[..., channel]
    header = volume.header.copy()  # type: ignore[no-untyped-call]
    output = nib.Nifti1Image(values, volume.affine, header)  # type: ignore[no-untyped-call]
    return gzip.compress(output.to_bytes())
