"""Project-scoped evaluation-set management."""

from fastapi import APIRouter, Depends

from monailabel.core.evaluation import (
    EvaluationSet,
    EvaluationSetCreate,
    EvaluationSetDelete,
    EvaluationSetExtend,
    EvaluationSetPublish,
    EvaluationSetUpdate,
    EvaluationSetVersion,
    ModelSplit,
)
from monailabel.server.access import Principal, Service, authorize

router = APIRouter(prefix="/api/projects/{project_id}", dependencies=[Depends(authorize)])


@router.get("/evaluation-sets")
def sets(project_id: str, service: Service) -> list[EvaluationSet]:
    return service.store.list(EvaluationSet, project_id)


@router.post("/evaluation-sets")
def create(project_id: str, body: EvaluationSetCreate, service: Service) -> EvaluationSet:
    return service.evaluation_sets.create(project_id, body)


@router.patch("/evaluation-sets/{set_id}")
def update(
    project_id: str, set_id: str, body: EvaluationSetUpdate, service: Service
) -> EvaluationSet:
    return service.evaluation_sets.update(project_id, set_id, body)


@router.post("/evaluation-sets/{set_id}/extend")
def extend(
    project_id: str, set_id: str, body: EvaluationSetExtend, service: Service
) -> EvaluationSet:
    return service.evaluation_sets.extend(project_id, set_id, body)


@router.delete("/evaluation-sets/{set_id}")
def delete(
    project_id: str, set_id: str, body: EvaluationSetDelete, service: Service
) -> dict[str, bool]:
    service.evaluation_sets.delete(project_id, set_id, body)
    return {"deleted": True}


@router.get("/evaluation-set-versions")
def versions(project_id: str, service: Service) -> list[EvaluationSetVersion]:
    return service.store.list(EvaluationSetVersion, project_id)


@router.post("/evaluation-sets/{set_id}/versions")
def publish(
    project_id: str, set_id: str, body: EvaluationSetPublish, service: Service, user: Principal
) -> EvaluationSetVersion:
    return service.evaluation_sets.publish(project_id, set_id, body, user.id)


@router.get("/model-splits")
def model_splits(project_id: str, service: Service) -> list[ModelSplit]:
    return service.store.list(ModelSplit, project_id)
