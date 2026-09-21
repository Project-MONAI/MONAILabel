"""Configured local Orthanc integration and durable source-series provenance."""

import io
import os

import httpx
import numpy as np
import pydicom

from monailabel.core.errors import Cancelled, Conflict, DomainError
from monailabel.core.models import Asset, DicomSeries, Job, JobStatus, Project, Split
from monailabel.dicom.nifti_view import viewing_series
from monailabel.server.dicom.imports import prepare_series
from monailabel.server.evaluation_sets import EvaluationSets
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.storage import Artifacts, Store


class Dicom:
    def __init__(self, store: Store, artifacts: Artifacts, jobs: Jobs):
        self.store, self.artifacts, self.jobs = store, artifacts, jobs
        self.url = os.environ.get("MONAILABEL_ORTHANC_URL", "http://127.0.0.1:8042").rstrip("/")

    def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        try:
            response = httpx.request(method, self.url + path, timeout=120, **kwargs)  # type: ignore[arg-type]
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise DomainError(
                "The configured DICOM server is unavailable or rejected the request."
            ) from exc

    def import_series(self, project_id: str, series_uid: str) -> Job:
        self.store.get(Project, project_id)
        if (
            not series_uid
            or any(c not in "0123456789." for c in series_uid)
            or len(series_uid) > 64
        ):
            raise DomainError("Provide a valid DICOM SeriesInstanceUID.")
        matches = self.request(
            "POST",
            "/tools/find",
            json={"Level": "Series", "Query": {"SeriesInstanceUID": series_uid}},
        ).json()
        if len(matches) != 1:
            raise DomainError(
                "The configured DICOM server must contain exactly one matching series."
            )

        def work(context: JobContext) -> Outcome:
            old = next(
                (s for s in self.store.list(DicomSeries, project_id) if s.series_uid == series_uid),
                None,
            )
            if old:
                return Outcome({"asset_id": old.asset_id})
            instances = self.request("GET", f"/series/{matches[0]}/instances").json()
            if not 1 <= len(instances) <= 3000:
                raise DomainError("Choose a scalar series with at most 3,000 instances.")
            contents = []
            total = 0
            for index, instance in enumerate(instances):
                context.progress(
                    index / len(instances), f"Reading DICOM slice {index + 1}/{len(instances)}"
                )
                content = self.request("GET", f"/instances/{instance['ID']}/file").content
                total += len(content)
                if total > 256 * 1024**2:
                    raise DomainError("DICOM series exceeds 256 MiB.")
                contents.append(content)
            asset, series = prepare_series(
                self.artifacts, project_id, self.url + "/dicom-web", contents
            )
            if series.series_uid != series_uid:
                raise DomainError("The downloaded series does not match the requested DICOM UID.")
            series = series.model_copy(update={"orthanc_series_id": matches[0]})
            with self.store.transaction() as session:
                if session.get(Job, context.job_id).status == JobStatus.CANCELLED:
                    raise Cancelled()
                sources = session.list(DicomSeries, project_id)
                duplicate = next((s for s in sources if s.series_uid == series_uid), None)
                if duplicate:
                    return Outcome(
                        {"asset_id": duplicate.asset_id, "dicom_series_id": duplicate.id}
                    )
                assets = session.list(Asset, project_id)
                same_patient = {
                    s.asset_id for s in sources if s.source_group_id == series.source_group_id
                }
                groups = {
                    a.group_id
                    for a in assets
                    if a.image_key == asset.image_key or a.id in same_patient
                }
                if len(groups) > 1:
                    raise Conflict("Matching images have conflicting source groups.")
                if groups:
                    asset = asset.model_copy(update={"group_id": groups.pop()})
                if any(a.group_id == asset.group_id and a.split != Split.POOL for a in assets):
                    raise Conflict("This patient already belongs to a learning split.")
                session.insert(asset)
                EvaluationSets.on_import(session, asset, is_new=True)
                session.insert(series)
            return Outcome({"asset_id": asset.id, "dicom_series_id": series.id})

        attempts = sum(
            j.kind == "dicom_import"
            and j.request.get("series_uid") == series_uid
            and j.status in {"failed", "cancelled", "interrupted"}
            for j in self.store.list(Job, project_id)
        )
        return self.jobs.submit(
            "dicom_import",
            project_id,
            {"series_uid": series_uid},
            work,
            key=f"dicom-import:{series_uid}:{attempts}",
        )

    def for_asset(self, asset_id: str) -> DicomSeries:
        asset = self.store.get(Asset, asset_id)
        series = next(
            (s for s in self.store.list(DicomSeries, asset.project_id) if s.asset_id == asset_id),
            None,
        )
        if not series:
            raise DomainError("This sample has no connected DICOM source series.")
        return series

    def ensure_view(self, asset_id: str, context: JobContext) -> DicomSeries:
        asset = self.store.get(Asset, asset_id)
        existing = next(
            (s for s in self.store.list(DicomSeries, asset.project_id) if s.asset_id == asset_id),
            None,
        )
        if existing:
            return existing
        if asset.kind != "volume3d" or not asset.affine:
            raise DomainError("Open 2D pathology images in QuPath. OHIF requires a 3D volume.")
        context.progress(0.35, "Preparing a DICOM viewing copy")
        contents = viewing_series(
            self.artifacts.array(asset.image_key)[..., 0],
            np.asarray(asset.affine),
            asset.name,
            asset.id,
            lambda value: context.progress(0.35 + value * 0.6, "Preparing a DICOM viewing copy"),
        )
        headers = [
            pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True) for content in contents
        ]
        source = DicomSeries(
            project_id=asset.project_id,
            asset_id=asset.id,
            study_uid=str(headers[0].StudyInstanceUID),
            series_uid=str(headers[0].SeriesInstanceUID),
            frame_of_reference_uid=str(headers[0].FrameOfReferenceUID),
            instance_uids=[str(header.SOPInstanceUID) for header in headers],
            source_keys=[self.artifacts.put(content) for content in contents],
            derived_from_nifti=True,
        )
        with self.store.transaction() as session:
            if session.get(Job, context.job_id).status == JobStatus.CANCELLED:
                raise Cancelled()
            existing = next(
                (s for s in session.list(DicomSeries, asset.project_id) if s.asset_id == asset_id),
                None,
            )
            if existing:
                return existing
            session.insert(source)
        return source
