"""Project-owned DICOMweb connections and resumable, per-series atomic imports."""

import base64
import json
from typing import Literal

from pydantic import JsonValue

from monailabel.core.dicom import (
    DicomConnection,
    DicomFilters,
    DicomImportSelection,
    DicomSearch,
)
from monailabel.core.errors import Cancelled, Conflict, DomainError
from monailabel.core.models import Asset, DicomSeries, Job, JobStatus, Project, Split
from monailabel.dicom.web import DicomwebClient, endpoint
from monailabel.server.dicom_imports import prepare_series
from monailabel.server.evaluation_sets import EvaluationSets
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.secrets import Secrets
from monailabel.server.storage import Artifacts, Store


class DicomConnections:
    def __init__(self, store: Store, artifacts: Artifacts, jobs: Jobs, secrets: Secrets):
        self.store, self.artifacts, self.jobs, self.secrets = store, artifacts, jobs, secrets

    @staticmethod
    def headers(authentication: str, username: str, password: str, token: str) -> dict[str, str]:
        if authentication == "basic":
            if not username or not password or ":" in username:
                raise DomainError("Provide a username and password for this server.")
            encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
            return {"Authorization": "Basic " + encoded}
        if authentication == "bearer":
            if not token.strip() or any(c in token for c in "\r\n"):
                raise DomainError("Provide a valid access token for this server.")
            return {"Authorization": "Bearer " + token.strip()}
        return {}

    def connect(
        self,
        project_id: str,
        name: str,
        url: str,
        authentication: Literal["none", "basic", "bearer"],
        username: str = "",
        password: str = "",
        token: str = "",
    ) -> DicomConnection:
        self.store.get(Project, project_id)
        url = endpoint(url)
        headers = self.headers(authentication, username, password, token)
        DicomwebClient(url, headers).check()
        credential = (
            self.secrets.save(project_id, f"DICOM · {name}", json.dumps(headers))
            if headers
            else None
        )
        connection = DicomConnection(
            project_id=project_id,
            name=name,
            url=url,
            authentication=authentication,
            credential_id=credential.id if credential else None,
        )
        with self.store.transaction() as session:
            session.insert(connection)
        return connection

    def get(self, project_id: str, identifier: str) -> DicomConnection:
        connection = self.store.get(DicomConnection, identifier)
        if connection.project_id != project_id:
            raise DomainError("This DICOM connection belongs to another project.", status=403)
        return connection

    def client(self, connection: DicomConnection) -> DicomwebClient:
        headers = (
            json.loads(self.secrets.resolve(connection.project_id, connection.credential_id))
            if connection.credential_id
            else {}
        )
        return DicomwebClient(connection.url, headers)

    def search(self, project_id: str, identifier: str, filters: DicomFilters) -> DicomSearch:
        connection = self.get(project_id, identifier)
        rows, truncated = self.client(connection).search(filters)
        existing = {
            (s.study_uid, s.series_uid): s.asset_id
            for s in self.store.list(DicomSeries, project_id)
        }
        result = []
        for row in rows:
            row = row.model_copy(
                update={"imported_asset_id": existing.get((row.study_uid, row.series_uid))}
            )
            if filters.import_status == "new" and row.imported_asset_id:
                continue
            if filters.import_status == "imported" and not row.imported_asset_id:
                continue
            result.append(row)
        result.sort(
            key=lambda row: (row.study_date, row.patient_id, row.series_description), reverse=True
        )
        return DicomSearch(items=result, truncated=truncated)

    def import_selection(
        self, project_id: str, identifier: str, selection: DicomImportSelection
    ) -> Job:
        connection = self.get(project_id, identifier)
        selected = list({(s.study_uid, s.series_uid): s for s in selection.series}.values())

        def work(context: JobContext) -> Outcome:
            client = self.client(connection)
            imported: list[JsonValue] = []
            skipped: list[JsonValue] = []
            failed: list[JsonValue] = []
            for index, ref in enumerate(selected):
                context.progress(
                    index / len(selected), f"Importing series {index + 1} of {len(selected)}"
                )
                try:
                    existing = next(
                        (
                            s
                            for s in self.store.list(DicomSeries, project_id)
                            if (s.study_uid, s.series_uid) == (ref.study_uid, ref.series_uid)
                        ),
                        None,
                    )
                    if existing:
                        skipped.append(existing.asset_id)
                    else:

                        def progress(part: float, index: int = index) -> None:
                            context.progress(
                                (index + part) / len(selected),
                                f"Downloading series {index + 1} of {len(selected)}",
                            )

                        contents = client.instances(ref, progress)
                        asset, series = prepare_series(
                            self.artifacts, project_id, connection.url, contents, connection.id
                        )
                        with self.store.transaction() as session:
                            job = session.get(Job, context.job_id)
                            if job.status == JobStatus.CANCELLED:
                                raise Cancelled()
                            duplicate = next(
                                (
                                    s
                                    for s in session.list(DicomSeries, project_id)
                                    if (s.study_uid, s.series_uid)
                                    == (ref.study_uid, ref.series_uid)
                                ),
                                None,
                            )
                            if duplicate:
                                skipped.append(duplicate.asset_id)
                            else:
                                assets = session.list(Asset, project_id)
                                same_patient = {
                                    s.asset_id
                                    for s in session.list(DicomSeries, project_id)
                                    if s.source_group_id == series.source_group_id
                                }
                                groups = {
                                    a.group_id
                                    for a in assets
                                    if a.image_key == asset.image_key or a.id in same_patient
                                }
                                if len(groups) > 1:
                                    raise Conflict(
                                        "Matching images or patient series have "
                                        "conflicting source groups."
                                    )
                                if groups:
                                    asset = asset.model_copy(update={"group_id": groups.pop()})
                                if any(
                                    a.group_id in {asset.group_id, series.study_uid}
                                    and a.split != Split.POOL
                                    for a in assets
                                ):
                                    raise Conflict(
                                        "This patient or study already belongs to a learning split."
                                    )
                                session.insert(asset)
                                asset = EvaluationSets.on_import(session, asset, is_new=True)
                                session.insert(series)
                                imported.append(asset.id)
                            session.update(
                                job.model_copy(
                                    update={
                                        "result": {
                                            "asset_ids": imported.copy(),
                                            "skipped_asset_ids": skipped.copy(),
                                            "failed": failed.copy(),
                                        }
                                    }
                                )
                            )
                except DomainError as exc:
                    failed.append({"series_uid": ref.series_uid, "error": str(exc)})
                with self.store.transaction() as session:
                    job = session.get(Job, context.job_id)
                    if job.status == JobStatus.CANCELLED:
                        raise Cancelled()
                    session.update(
                        job.model_copy(
                            update={
                                "result": {
                                    "asset_ids": imported.copy(),
                                    "skipped_asset_ids": skipped.copy(),
                                    "failed": failed.copy(),
                                }
                            }
                        )
                    )
            return Outcome({"asset_ids": imported, "skipped_asset_ids": skipped, "failed": failed})

        return self.jobs.submit(
            "dicom_import",
            project_id,
            {
                "connection_id": identifier,
                "series": [ref.model_dump(mode="json") for ref in selected],
            },
            work,
        )
