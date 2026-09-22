"""Manager-only discovery and import of hosted annotation models."""

from fastapi import APIRouter, Depends

from monailabel.core.models import ModelRecord
from monailabel.server.access import Service, authorize
from monailabel.server.models.catalog import (
    CatalogRequest,
    ImportModelRequest,
    ModelCatalog,
    ProviderChangeRequest,
    ServiceInfo,
)

router = APIRouter(prefix="/api/projects/{project_id}", dependencies=[Depends(authorize)])


@router.get("/model-services")
def services(project_id: str, service: Service) -> list[ServiceInfo]:
    return service.model_catalogs.services(project_id)


@router.post("/model-catalog")
def catalog(project_id: str, body: CatalogRequest, service: Service) -> ModelCatalog:
    return service.model_catalogs.list_models(project_id, body)


@router.post("/model-import", status_code=201)
def import_model(project_id: str, body: ImportModelRequest, service: Service) -> ModelRecord:
    return service.model_catalogs.import_model(project_id, body)


@router.get("/models/{model_id}/model-services")
def preset_services(project_id: str, model_id: str, service: Service) -> list[ServiceInfo]:
    return service.model_catalogs.preset_services(project_id, model_id)


@router.post("/models/{model_id}/model-catalog")
def preset_catalog(
    project_id: str, model_id: str, body: CatalogRequest, service: Service
) -> ModelCatalog:
    return service.model_catalogs.preset_catalog(project_id, model_id, body)


@router.post("/models/{model_id}/provider")
def change_provider(
    project_id: str, model_id: str, body: ProviderChangeRequest, service: Service
) -> ModelRecord:
    return service.model_catalogs.change_provider(project_id, model_id, body)
