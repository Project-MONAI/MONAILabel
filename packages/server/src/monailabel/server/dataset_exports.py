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

"""Snapshot selected samples and stream their saved annotations into a downloadable ZIP."""

import hashlib
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from pydantic import Field

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    Annotation,
    Asset,
    Contract,
    Job,
    JobStatus,
    Project,
    ReviewDecision,
)
from monailabel.core.review_units import ReviewUnit, UnitAnnotation
from monailabel.core.video import TrackAnnotation, VideoAsset
from monailabel.server.data import nifti_bytes
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.storage import Artifacts, Session, Store


class ExportRequest(Contract):
    asset_ids: list[str] = Field(min_length=1, max_length=10000)
    annotations: Literal["latest", "all"] = "latest"


def filename(name: str) -> str:
    """Keep archive members relative, portable and independent of user-supplied paths."""
    return (
        re.sub(r"[^a-zA-Z0-9._-]", "_", name.replace("\\", "/").rsplit("/", 1)[-1]).strip(".")[:160]
        or "image"
    )


class DatasetExports:
    def __init__(self, store: Store, artifacts: Artifacts, jobs: Jobs):
        self.store, self.artifacts, self.jobs = store, artifacts, jobs

    def start(self, project_id: str, request: ExportRequest, key: str | None = None) -> Job:
        identifiers = list(dict.fromkeys(request.asset_ids))
        snapshot: dict[str, Any] = {}

        def capture(session: Session) -> None:
            project = session.get(Project, project_id)
            videos = {a.id: a for a in session.list(VideoAsset, project_id)}
            assets = [videos[i] if i in videos else session.get(Asset, i) for i in identifiers]
            if any(a.project_id != project_id for a in assets):
                raise DomainError("Every exported file must belong to this project.", status=403)
            records: list[Annotation | TrackAnnotation] = [
                *session.list(Annotation, project_id),
                *session.list(TrackAnnotation, project_id),
            ]
            snapshot.update(
                project=project,
                assets=assets,
                annotations={
                    a.id: [
                        r
                        for r in records
                        if r.asset_id == a.id
                        and (request.annotations == "all" or r.id == a.annotation_id)
                    ]
                    for a in assets
                },
                decisions=session.list(ReviewDecision, project_id),
                units=[
                    u for u in session.list(ReviewUnit, project_id) if u.asset_id in identifiers
                ],
                unit_annotations=[
                    u for u in session.list(UnitAnnotation, project_id) if u.asset_id in identifiers
                ],
                created_at=datetime.now(UTC).isoformat(),
            )

        return self.jobs.submit(
            "dataset_export",
            project_id,
            request.model_dump(mode="json"),
            lambda context: self.build(context, snapshot, request.annotations),
            key,
            guard=capture,
        )

    def build(self, context: JobContext, snapshot: dict[str, Any], mode: str) -> Outcome:
        project = snapshot["project"]
        assets = snapshot["assets"]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at": snapshot["created_at"],
            "project": {"id": project.id, "name": project.name, "version": project.version},
            "classes": [label.model_dump(mode="json") for label in project.labels],
            "annotation_versions": mode,
            "conventions": {
                "masks": "Class IDs; 0 is background. Missing masks mean no saved annotation.",
                "volumes": "NIfTI masks use the source grid; affine maps IJK to RAS.",
                "images": "PNG masks use rows, columns in the source image pixel grid.",
                "video": "Track JSON uses zero-based frames and source pixel coordinates.",
                "coverage": "Coverage is limited to covered_labels and specified regions/ranges.",
                "versions": "Saved submissions at export start; excludes drafts and proposals.",
            },
            "assets": [],
            "files": [],
        }
        annotation_count = sum(len(snapshot["annotations"][a.id]) for a in assets)
        total = (
            len(assets) + annotation_count + sum(len(self.units(snapshot, a, mode)) for a in assets)
        )
        completed = 0
        with tempfile.TemporaryDirectory(prefix="export-", dir=self.artifacts.root) as temporary:
            path = Path(temporary) / "dataset.zip"
            with ZipFile(
                path, "w", compression=ZIP_DEFLATED, compresslevel=3, allowZip64=True
            ) as archive:
                for asset in assets:
                    context.check_cancelled()
                    folder = f"samples/{asset.id}"
                    metadata = asset.model_dump(
                        mode="json",
                        exclude={"image_key", "source_key", "fixture_key", "metadata_key"},
                    )
                    metadata["annotations"] = []
                    if asset.source_key:
                        source_path = self.artifacts.path(asset.source_key)
                        source_name = f"{folder}/image/{filename(asset.name)}"
                    else:
                        source_path = Path(temporary) / "image.npy"
                        if isinstance(asset, Asset) and asset.affine:
                            source_path = source_path.with_suffix(".nii")
                            source_path.write_bytes(
                                nifti_bytes(
                                    self.artifacts.array(asset.image_key)[..., 0], asset.affine
                                )
                            )
                        else:
                            np.save(
                                source_path,
                                self.artifacts.array(asset.image_key),
                                allow_pickle=False,
                            )
                        source_name = f"{folder}/image/{source_path.name}"
                    self.add_file(
                        archive,
                        source_path,
                        source_name,
                        context,
                        manifest,
                        completed / total,
                        1 / total,
                    )
                    completed += 1
                    metadata["image"] = source_name
                    for annotation in snapshot["annotations"][asset.id]:
                        context.progress(
                            0.9 * completed / total,
                            f"Preparing {asset.name}: revision {annotation.revision}",
                        )
                        version = annotation.model_dump(
                            mode="json",
                            exclude={"mask_key", "tracks_key", "project_id", "asset_id"},
                        )
                        version["review"] = self.review(snapshot, annotation.id)
                        if isinstance(annotation, TrackAnnotation):
                            mask_path = self.artifacts.path(annotation.tracks_key)
                            suffix = ".json"
                        else:
                            mask = self.artifacts.array(annotation.mask_key)
                            mask_path = Path(temporary) / "mask"
                            if asset.affine:
                                source = (
                                    self.artifacts.read(asset.source_key)
                                    if asset.source_key
                                    else None
                                )
                                mask_path.write_bytes(nifti_bytes(mask, asset.affine, source))
                                suffix = ".nii"
                            else:
                                Image.fromarray(mask).save(mask_path, format="PNG")
                                suffix = ".png"
                        name = f"{folder}/labels/revision-{annotation.revision:06d}{suffix}"
                        self.add_file(
                            archive,
                            mask_path,
                            name,
                            context,
                            manifest,
                            completed / total,
                            1 / total,
                        )
                        completed += 1
                        version["file"] = name
                        metadata["annotations"].append(version)
                    metadata["review_units"] = self.units(snapshot, asset, mode)
                    unit_records = {u.id: u for u in snapshot["unit_annotations"]}
                    for reference in metadata["review_units"]:
                        unit = unit_records[reference["id"]]
                        if unit.scope.kind == "frames":
                            unit_path = self.artifacts.path(unit.mask_key)
                            suffix = ".json"
                        else:
                            unit_path = Path(temporary) / "reference.png"
                            Image.fromarray(self.artifacts.array(unit.mask_key)).save(
                                unit_path, format="PNG"
                            )
                            suffix = ".png"
                        reference["file"] = (
                            f"{folder}/review-units/{unit.unit_id}/revision-{unit.revision:06d}{suffix}"
                        )
                        self.add_file(
                            archive,
                            unit_path,
                            reference["file"],
                            context,
                            manifest,
                            completed / total,
                            1 / total,
                        )
                        completed += 1
                    if isinstance(asset, VideoAsset):
                        metadata["video_metadata"] = json.loads(
                            self.artifacts.read(asset.metadata_key)
                        )
                    manifest["assets"].append(metadata)
                context.progress(0.92, "Writing manifest")
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
                )
            context.progress(0.95, "Saving ZIP for download")
            artifact = self.artifacts.put_file(path)
            context.check_cancelled()
            return Outcome(
                {
                    "archive_key": artifact,
                    "filename": filename(project.name) + "-dataset.zip",
                    "asset_count": len(assets),
                    "annotation_count": annotation_count,
                    "size_bytes": path.stat().st_size,
                }
            )

    @staticmethod
    def review(snapshot: dict[str, Any], annotation_id: str) -> dict[str, Any]:
        decisions = [d for d in snapshot["decisions"] if d.annotation_id == annotation_id]
        return decisions[-1].model_dump(mode="json") if decisions else {"verdict": "pending"}

    def units(
        self, snapshot: dict[str, Any], asset: Asset | VideoAsset, mode: str
    ) -> list[dict[str, Any]]:
        units = {u.id: u for u in snapshot["units"] if u.asset_id == asset.id}
        return [
            {
                **u.model_dump(
                    mode="json", exclude={"image_key", "mask_key", "project_id", "asset_id"}
                ),
                "name": units[u.unit_id].name,
                "review": self.review(snapshot, u.id),
            }
            for u in snapshot["unit_annotations"]
            if u.unit_id in units and (mode == "all" or u.id == units[u.unit_id].annotation_id)
        ]

    @staticmethod
    def add_file(
        archive: ZipFile,
        path: Path,
        name: str,
        context: JobContext,
        manifest: dict[str, Any],
        start: float,
        fraction: float,
    ) -> None:
        size, copied = path.stat().st_size, 0
        digest = hashlib.sha256()
        with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
            while block := source.read(1024 * 1024):
                context.check_cancelled()
                target.write(block)
                digest.update(block)
                copied += len(block)
                context.progress(0.9 * (start + fraction * copied / max(size, 1)), f"Adding {name}")
        manifest["files"].append({"path": name, "size_bytes": size, "sha256": digest.hexdigest()})

    def download(self, job_id: str) -> tuple[Path, str]:
        job = self.store.get(Job, job_id)
        if job.kind != "dataset_export" or job.status != JobStatus.SUCCEEDED:
            raise DomainError("This export is not ready for download.", status=409)
        return self.artifacts.path(str(job.result["archive_key"])), str(job.result["filename"])
