"""Authenticated review items and bounded binary region submissions."""

import numpy as np
from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.types import Message

from monailabel.core.errors import DomainError
from monailabel.core.models import Annotation, Asset, ReviewDecision, ReviewRequest
from monailabel.core.review_units import ReviewUnit, UnitAnnotation, UnitDecision
from monailabel.server.access import Principal, Service, authorize
from monailabel.server.review_units import ReviewUnits

router = APIRouter(prefix="/api", dependencies=[Depends(authorize)])


@router.get("/projects/{project_id}/review-units")
def units(project_id: str, service: Service) -> list[ReviewUnit]:
    return service.store.list(ReviewUnit, project_id)


@router.get("/review-units/{unit_id}")
def unit(unit_id: str, service: Service) -> ReviewUnit:
    return service.store.get(ReviewUnit, unit_id)


@router.get("/review-units/{unit_id}/preview")
def preview(
    unit_id: str,
    service: Service,
    base_revision: int = Query(ge=1),
    frame: int | None = Query(default=None, ge=0),
    overlay: bool = True,
) -> Response:
    from monailabel.server.review_units.preview import render

    content = render(
        service.store,
        service.artifacts,
        service.store.get(ReviewUnit, unit_id),
        base_revision,
        frame,
        overlay,
    )
    return Response(content, media_type="image/png", headers={"Cache-Control": "private, no-store"})


@router.get("/review-units/{unit_id}/annotations")
def annotations(unit_id: str, service: Service) -> list[UnitAnnotation]:
    record = service.store.get(ReviewUnit, unit_id)
    return [
        a for a in service.store.list(UnitAnnotation, record.project_id) if a.unit_id == unit_id
    ]


@router.post("/review-units/{unit_id}/decision", status_code=201)
def decide(unit_id: str, body: UnitDecision, service: Service, user: Principal) -> ReviewDecision:
    return ReviewUnits(service.store, service.artifacts).decide(unit_id, body, user)


@router.post("/assets/{asset_id}/region-submissions", status_code=201)
async def submit_regions(
    asset_id: str, request: Request, service: Service, user: Principal
) -> Annotation:
    asset = service.store.get(Asset, asset_id)
    expected = int(np.prod(asset.spatial_shape))
    size = 0

    async def receive() -> Message:
        nonlocal size
        message = await request.receive()
        size += len(message.get("body", b""))
        if size > expected + 8 * 1024 * 1024:
            raise DomainError(
                "Region submission exceeds its source mask and metadata limits.", status=413
            )
        return message

    bounded = Request(request.scope, receive=receive)
    async with bounded.form(max_files=1, max_fields=1, max_part_size=8 * 1024 * 1024) as form:
        metadata, upload = form.get("metadata"), form.get("mask")
        if not isinstance(metadata, str) or not isinstance(upload, UploadFile):
            raise DomainError("Provide the region metadata and source-grid binary mask.")
        try:
            body = ReviewRequest.model_validate_json(metadata)
        except ValidationError as exc:
            raise DomainError("Invalid region submission metadata.") from exc
        if not body.regions or body.mask is not None:
            raise DomainError("Choose regions and provide the mask as a binary part.")
        content = await upload.read(expected + 1)
        if len(content) != expected:
            raise DomainError("The mask must match the full source image geometry.")
        mask = np.frombuffer(content, dtype=np.uint8).reshape(asset.spatial_shape)
        return await run_in_threadpool(
            service.annotations.review,
            asset_id,
            body.model_copy(update={"reviewer": user.id}),
            mask,
        )
