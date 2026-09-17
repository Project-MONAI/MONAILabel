"""Project import, source metadata, and authenticated DICOMweb reads for OHIF."""

from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import Field, SecretStr

from monailabel.core.dicom import DicomConnection, DicomFilters, DicomImportSelection, DicomSearch
from monailabel.core.models import Contract, DicomSeries, Job
from monailabel.server.access import Principal, Service, authorize
from monailabel.server.dicom_read import dicom_response

router = APIRouter(prefix="/api", dependencies=[Depends(authorize)])


class ConnectDicom(Contract):
    name: str = Field(min_length=1, max_length=120)
    url: str = Field(min_length=1, max_length=2000)
    authentication: Literal["none", "basic", "bearer"] = "none"
    username: str = Field(default="", max_length=200)
    password: SecretStr = SecretStr("")
    token: SecretStr = SecretStr("")


class Connections(Contract):
    connections: list[DicomConnection]
    suggested_url: str


@router.get("/projects/{project_id}/dicom-connections")
def connections(project_id: str, service: Service) -> Connections:
    return Connections(
        connections=service.store.list(DicomConnection, project_id),
        suggested_url=service.dicom.url + "/dicom-web",
    )


@router.post("/projects/{project_id}/dicom-connections", status_code=201)
def connect(project_id: str, body: ConnectDicom, service: Service) -> DicomConnection:
    return service.dicom_connections.connect(
        project_id,
        body.name,
        body.url,
        body.authentication,
        body.username,
        body.password.get_secret_value(),
        body.token.get_secret_value(),
    )


@router.post("/projects/{project_id}/dicom-connections/{connection_id}/connect")
def reconnect(project_id: str, connection_id: str, service: Service) -> DicomConnection:
    connection = service.dicom_connections.get(project_id, connection_id)
    service.dicom_connections.client(connection).check()
    return connection


@router.post("/projects/{project_id}/dicom-connections/{connection_id}/search")
def search(
    project_id: str, connection_id: str, body: DicomFilters, service: Service
) -> DicomSearch:
    return service.dicom_connections.search(project_id, connection_id, body)


@router.post("/projects/{project_id}/dicom-connections/{connection_id}/imports", status_code=202)
def import_selection(
    project_id: str,
    connection_id: str,
    body: DicomImportSelection,
    service: Service,
) -> Job:
    return service.dicom_connections.import_selection(project_id, connection_id, body)


class ImportSeries(Contract):
    series_uid: str = Field(min_length=1, max_length=64)


@router.post("/projects/{project_id}/dicom-series", status_code=202)
def import_series(project_id: str, body: ImportSeries, service: Service) -> Job:
    return service.dicom.import_series(project_id, body.series_uid)


@router.get("/projects/{project_id}/dicom-series")
def project_series(project_id: str, service: Service) -> list[DicomSeries]:
    return service.store.list(DicomSeries, project_id)


@router.get("/assets/{asset_id}/dicom")
def asset_series(asset_id: str, service: Service) -> DicomSeries:
    return service.dicom.for_asset(asset_id)


@router.get("/assets/{asset_id}/dicomweb/{path:path}")
def asset_dicomweb(asset_id: str, path: str, request: Request, service: Service) -> Response:
    # The viewer edits this one asset. Related-study searches must stay in that scope,
    # even when derived files in other projects reuse synthetic patient metadata.
    source = service.dicom.for_asset(asset_id)
    return dicom_response(path, dict(request.query_params), [source], service.artifacts)


@router.get("/dicomweb/{path:path}")
def dicomweb(path: str, request: Request, service: Service, user: Principal) -> Response:
    all_series = service.store.list(DicomSeries)
    allowed = [s for s in all_series if service.auth.roles(user, s.project_id)]
    return dicom_response(path, dict(request.query_params), allowed, service.artifacts)
