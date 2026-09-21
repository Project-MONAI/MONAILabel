"""Versioned HTTP transport. Business rules live in application services."""

import gzip
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import Field, JsonValue, ValidationError
from starlette.concurrency import run_in_threadpool

from monailabel.core.dataset_templates import DatasetTemplate, DatasetTemplateImport
from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
    TERMINAL_STATUSES,
    AnnotateRequest,
    Annotation,
    Asset,
    AssistantReply,
    AssistantRequest,
    BatchReviewRequest,
    ClassificationProposal,
    CompleteReview,
    Contract,
    DecisionRequest,
    DeleteAssetsRequest,
    DeleteModelRequest,
    DeleteProjectRequest,
    DeletionResult,
    EvaluateRequest,
    Evaluation,
    ImageImport,
    ImageMetadata,
    Job,
    JobLogPage,
    Learner,
    LearnerCreate,
    LearnerUpdate,
    ModelRecord,
    ModelRegister,
    ModelUpdate,
    Project,
    ProjectCreate,
    ProjectUpdate,
    PromoteRequest,
    Promotion,
    Proposal,
    RecipeInfo,
    RegionProposal,
    RestoreRequest,
    ReviewDecision,
    ReviewRequest,
    Role,
    SelectionRequest,
    Snapshot,
    SnapshotRequest,
    Split,
    StartTraining,
    TrainingReport,
    TrainingSource,
    TrainRequest,
)
from monailabel.server.access import Principal, Service, authorize
from monailabel.server.data import decode_image, nifti_bytes
from monailabel.server.demo import create_demo
from monailabel.server.training_samples import training_sources
from monailabel.server.uploads import read_upload

router = APIRouter(prefix="/api", dependencies=[Depends(authorize)])
Idempotency = Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)]


@router.get("/classification-proposals/{classification_id}")
def classification_proposal(classification_id: str, service: Service) -> ClassificationProposal:
    return service.store.get(ClassificationProposal, classification_id)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0", "mode": "local"}


@router.post("/projects", status_code=201)
def create_project(body: ProjectCreate, service: Service, user: Principal) -> Project:
    project = service.datasets.create(body)
    service.auth.add_member(project.id, user.id, [Role.MANAGER])
    service.presets.ensure(project.id)
    return service.store.get(Project, project.id)


@router.get("/projects")
def projects(service: Service, user: Principal) -> list[Project]:
    return [p for p in service.store.list(Project) if service.auth.roles(user, p.id)]


@router.get("/projects/{project_id}")
def project(project_id: str, service: Service) -> Project:
    return service.store.get(Project, project_id)


@router.get("/projects/{project_id}/changes")
def project_changes(project_id: str, service: Service) -> dict[str, int]:
    service.store.get(Project, project_id)
    return {"version": service.store.change_version(project_id)}


@router.patch("/projects/{project_id}")
def update_project(project_id: str, body: ProjectUpdate, service: Service) -> Project:
    return service.datasets.update(project_id, body)


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, body: DeleteProjectRequest, service: Service) -> DeletionResult:
    return service.deletion.project(project_id, body.confirmation_name)


@router.delete("/projects/{project_id}/assets")
def delete_assets(project_id: str, body: DeleteAssetsRequest, service: Service) -> DeletionResult:
    return service.deletion.assets(project_id, body.asset_ids)


@router.delete("/assets/{asset_id}")
def delete_asset(asset_id: str, service: Service) -> DeletionResult:
    asset = service.store.get(Asset, asset_id)
    return service.deletion.assets(asset.project_id, [asset_id])


@router.post("/demo", status_code=201)
def demo(service: Service, user: Principal) -> dict[str, str]:
    result = create_demo(service.store, service.artifacts, service.datasets)
    service.auth.add_member(result["project_id"], user.id, [Role.MANAGER])
    return result


@router.post("/projects/{project_id}/assets", status_code=201)
def import_image(project_id: str, body: ImageImport, service: Service) -> Asset:
    return service.datasets.import_image(project_id, body)


@router.post("/projects/{project_id}/assets/upload", status_code=201)
async def upload_image(
    project_id: str,
    request: Request,
    metadata: Annotated[ImageMetadata, Query()],
    service: Service,
) -> Asset:
    content = await read_upload(request)
    return await run_in_threadpool(service.datasets.import_content, project_id, metadata, content)


@router.get("/projects/{project_id}/assets")
def assets(project_id: str, service: Service) -> list[Asset]:
    service.store.get(Project, project_id)
    return service.store.list(Asset, project_id)


@router.get("/projects/{project_id}/training-sources")
def list_training_sources(project_id: str, service: Service) -> list[TrainingSource]:
    with service.store.transaction() as session:
        session.get(Project, project_id)
        return training_sources(session, project_id)


@router.get("/assets/{asset_id}")
def asset(asset_id: str, service: Service) -> Asset:
    return service.store.get(Asset, asset_id)


@router.get("/assets/{asset_id}/image")
def image(asset_id: str, service: Service) -> Response:
    asset = service.store.get(Asset, asset_id)
    if asset.kind == "volume3d" and asset.affine:
        if asset.source_key:
            content = service.artifacts.read(asset.source_key)
            if content.startswith(b"\x1f\x8b"):
                content = gzip.decompress(content)
        else:
            content = nifti_bytes(service.artifacts.array(asset.image_key)[..., 0], asset.affine)
        filename = f"{asset.id}.nii"
    elif asset.source_key:
        content = service.artifacts.read(asset.source_key)
        filename = f"{asset.id}{Path(asset.name).suffix.lower()}"
    else:
        raise DomainError("No source image is available.")
    return Response(
        content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/assets/{asset_id}/fixture")
def fixture(asset_id: str, service: Service) -> dict[str, JsonValue]:
    asset = service.store.get(Asset, asset_id)
    project = service.store.get(Project, asset.project_id)
    if not project.is_demo or not asset.fixture_key:
        raise DomainError("Only synthetic demo assets have fixture reference masks.", status=404)
    return {
        "mask": service.artifacts.array(asset.fixture_key).tolist(),
        "synthetic_reference": True,
    }


@router.post("/assets/{asset_id}/assign-train")
def assign_train(asset_id: str, service: Service) -> Asset:
    return service.datasets.assign_split(asset_id, Split.TRAIN)


@router.post("/assets/{asset_id}/assign-validation")
def assign_validation(asset_id: str, service: Service) -> Asset:
    return service.datasets.assign_split(asset_id, Split.VALIDATION)


@router.post("/assets/{asset_id}/annotate", status_code=202)
def annotate(
    asset_id: str, body: AnnotateRequest, service: Service, key: Idempotency = None
) -> Job:
    return service.annotations.annotate(asset_id, body, key)


@router.get("/proposals/{proposal_id}")
def proposal(proposal_id: str, service: Service) -> Proposal:
    return service.store.get(Proposal, proposal_id)


@router.get("/proposals/{proposal_id}/mask")
def proposal_mask(proposal_id: str, service: Service) -> dict[str, JsonValue]:
    proposal = service.store.get(Proposal, proposal_id)
    return {"mask": service.artifacts.array(proposal.mask_key).tolist()}


@router.get("/proposals/{proposal_id}/mask.bin")
def proposal_mask_binary(proposal_id: str, service: Service) -> Response:
    proposal = service.store.get(Proposal, proposal_id)
    return Response(
        service.artifacts.array(proposal.mask_key).tobytes(), media_type="application/octet-stream"
    )


@router.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str, service: Service) -> Proposal:
    return service.annotations.reject(proposal_id)


@router.post("/assets/{asset_id}/review", status_code=201)
def review(asset_id: str, body: ReviewRequest, service: Service, user: Principal) -> Annotation:
    return service.annotations.review(asset_id, body.model_copy(update={"reviewer": user.id}))


@router.post("/assets/{asset_id}/review-mask", status_code=201)
async def review_binary(
    asset_id: str,
    request: Request,
    service: Service,
    user: Principal,
    base_revision: Annotated[int, Query(ge=0)],
    covered_labels: Annotated[list[int], Query(min_length=1, max_length=32)],
    proposal_id: str | None = None,
) -> Annotation:
    asset = service.store.get(Asset, asset_id)
    content = await read_upload(request)
    if len(content) != int(np.prod(asset.spatial_shape)):
        raise DomainError("Binary mask must have one uint8 value per source voxel in IJK order.")
    mask = np.frombuffer(content, dtype=np.uint8).reshape(asset.spatial_shape)
    body = ReviewRequest(
        base_revision=base_revision,
        proposal_id=proposal_id,
        covered_labels=covered_labels,
        reviewer=user.id,
    )
    return await run_in_threadpool(service.annotations.review, asset_id, body, mask)


@router.post("/assets/{asset_id}/review-complete", status_code=201)
async def complete_review(
    asset_id: str, request: Request, service: Service, user: Principal
) -> Annotation:
    # JSON metadata in a header keeps free-text comments out of request URLs/access logs.
    try:
        metadata = CompleteReview.model_validate_json(
            request.headers.get("X-MONAILABEL-REVIEW", "")
        )
    except ValidationError:
        raise DomainError("Provide valid review revision, coverage and comment metadata.") from None
    asset = service.store.get(Asset, asset_id)
    content = await read_upload(request)
    if len(content) != int(np.prod(asset.spatial_shape)):
        raise DomainError("Binary mask must have one uint8 value per source voxel in IJK order.")
    mask = np.frombuffer(content, dtype=np.uint8).reshape(asset.spatial_shape)
    body = ReviewRequest(
        base_revision=metadata.base_revision,
        covered_labels=metadata.covered_labels,
        reviewer=user.id,
    )
    return await run_in_threadpool(
        service.annotations.review,
        asset_id,
        body,
        mask,
        DecisionRequest(verdict="accepted", comment=metadata.comment),
    )


@router.post("/assets/{asset_id}/restore", status_code=201)
def restore(asset_id: str, body: RestoreRequest, service: Service, user: Principal) -> Annotation:
    return service.annotations.restore(asset_id, body.model_copy(update={"reviewer": user.id}))


@router.get("/assets/{asset_id}/annotations")
def annotations(asset_id: str, service: Service) -> list[Annotation]:
    asset = service.store.get(Asset, asset_id)
    return [a for a in service.store.list(Annotation, asset.project_id) if a.asset_id == asset_id]


@router.post("/assets/{asset_id}/mask-import", status_code=201)
async def import_mask(
    asset_id: str,
    request: Request,
    service: Service,
    user: Principal,
    name: str,
    base_revision: int = Query(ge=0),
) -> Annotation:
    asset = service.store.get(Asset, asset_id)
    project = service.store.get(Project, asset.project_id)
    if asset.kind != "volume3d" or not name.lower().endswith((".nii", ".nii.gz")):
        raise DomainError("Reference mask import requires a NIfTI label map and volume.")
    content = await read_upload(request)
    values, affine = await run_in_threadpool(decode_image, name, content)
    if (
        affine is None
        or asset.affine is None
        or not np.allclose(affine, asset.affine, rtol=1e-5, atol=1e-4)
    ):
        raise DomainError(
            "Mask affine does not match the image; resample explicitly before import."
        )
    if not set(np.unique(values)) <= {label.id for label in project.labels}:
        raise DomainError("Reference mask contains unknown label IDs.")
    return service.annotations.review(
        asset_id,
        ReviewRequest(
            base_revision=base_revision,
            reviewer=user.id,
            covered_labels=[label.id for label in project.labels],
        ),
        imported_mask=values[..., 0].astype(np.uint8),
    )


@router.get("/annotations/{annotation_id}/mask")
def annotation_mask(annotation_id: str, service: Service) -> dict[str, JsonValue]:
    annotation = service.store.get(Annotation, annotation_id)
    return {"mask": service.artifacts.array(annotation.mask_key).tolist()}


@router.get("/annotations/{annotation_id}/mask.bin")
def annotation_mask_binary(annotation_id: str, service: Service) -> Response:
    annotation = service.store.get(Annotation, annotation_id)
    return Response(
        service.artifacts.array(annotation.mask_key).tobytes(),
        media_type="application/octet-stream",
    )


@router.get("/assets/{asset_id}/segmentation.nii")
def export_segmentation(asset_id: str, service: Service) -> Response:
    asset = service.store.get(Asset, asset_id)
    if not asset.affine or not asset.annotation_id:
        raise DomainError("Export requires a reviewed NIfTI asset.")
    annotation = service.store.get(Annotation, asset.annotation_id)
    source = service.artifacts.read(asset.source_key) if asset.source_key else None
    content = nifti_bytes(service.artifacts.array(annotation.mask_key), asset.affine, source)
    return Response(content, media_type="application/octet-stream")


@router.get("/projects/{project_id}/models")
def models(project_id: str, service: Service, asset_id: str | None = None) -> list[ModelRecord]:
    return service.models.available(project_id, asset_id)


@router.post("/projects/{project_id}/models", status_code=201)
def register_model(project_id: str, body: ModelRegister, service: Service) -> ModelRecord:
    return service.models.register(project_id, body)


class DefaultModelRequest(Contract):
    model_id: str
    base_version: int = Field(ge=0)


@router.put("/projects/{project_id}/annotation-model")
def set_annotation_model(project_id: str, body: DefaultModelRequest, service: Service) -> Project:
    return service.models.set_default(project_id, body.model_id, body.base_version)


class DefaultsRequest(Contract):
    model_id: str
    label_ids: list[int] = Field(min_length=1)
    base_version: int = Field(ge=0)


@router.put("/projects/{project_id}/defaults")
def set_defaults(project_id: str, body: DefaultsRequest, service: Service) -> Project:
    model = service.models.get(project_id, body.model_id)
    with service.store.transaction() as session:
        project = session.get(Project, project_id)
        supported = (
            {label.id for label in project.labels}
            if service.models.promptable(model)
            else set(model.label_ids)
        ) - {0}
        if not set(body.label_ids) <= supported:
            raise DomainError("The selected model does not support these foreground labels.")
        if model.read_only or model.inherit_targets:
            service.models.validate_targets(
                model, [label.name for label in project.labels if label.id in body.label_ids]
            )
        if project.version != body.base_version:
            raise Conflict("Project defaults changed. Reload before selecting defaults.")
        project = project.model_copy(
            update={
                "defaults": project.defaults | {label: model.id for label in body.label_ids},
                "version": project.version + 1,
            }
        )
        session.update(project)
        return project


@router.post("/projects/{project_id}/snapshots", status_code=201)
def create_snapshot(
    project_id: str, service: Service, user: Principal, body: SnapshotRequest | None = None
) -> Snapshot:
    return service.datasets.snapshot(project_id, request=body, authorized_by=user.id)


@router.get("/projects/{project_id}/snapshots")
def snapshots(project_id: str, service: Service) -> list[Snapshot]:
    return service.store.list(Snapshot, project_id)


@router.post("/projects/{project_id}/train", status_code=202)
def train(project_id: str, body: TrainRequest, service: Service, key: Idempotency = None) -> Job:
    return service.learning.train(project_id, body, key)


@router.post("/projects/{project_id}/evaluate", status_code=202)
def evaluate(
    project_id: str,
    body: EvaluateRequest,
    service: Service,
    user: Principal,
    key: Idempotency = None,
) -> Job:
    return service.learning.evaluate(project_id, body, key, authorized_by=user.id)


@router.get("/evaluations/{evaluation_id}")
def evaluation(evaluation_id: str, service: Service) -> Evaluation:
    return service.store.get(Evaluation, evaluation_id)


@router.get("/projects/{project_id}/evaluations")
def evaluations(project_id: str, service: Service) -> list[Evaluation]:
    return service.store.list(Evaluation, project_id)


@router.post("/projects/{project_id}/promote", status_code=201)
def promote(project_id: str, body: PromoteRequest, service: Service) -> Promotion:
    return service.learning.promote(project_id, body)


@router.post("/promotions/{promotion_id}/rollback")
def rollback(promotion_id: str, service: Service) -> Project:
    return service.learning.rollback(promotion_id)


@router.post("/projects/{project_id}/select", status_code=202)
def select(project_id: str, body: SelectionRequest, service: Service) -> Job:
    return service.selection.select(project_id, body)


@router.post("/projects/{project_id}/assistant")
def assistant(
    project_id: str, body: AssistantRequest, service: Service, user: Principal
) -> AssistantReply:
    return service.assistants.run(project_id, body, user)


@router.get("/projects/{project_id}/jobs")
def jobs(project_id: str, service: Service) -> list[Job]:
    return service.store.list(Job, project_id)


@router.get("/jobs/{job_id}")
def job(job_id: str, service: Service) -> Job:
    return service.store.get(Job, job_id)


@router.get("/jobs/{job_id}/logs")
def job_logs(
    job_id: str,
    service: Service,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=1000),
) -> JobLogPage:
    return service.jobs.logs(job_id, after, limit)


@router.get("/jobs/{job_id}/logs/download")
def download_job_log(job_id: str, service: Service) -> StreamingResponse:
    job = service.store.get(Job, job_id)
    kind = (
        "evaluation"
        if job.kind in {"evaluate", "training_report"}
        else "batch-segmentation"
        if job.kind == "batch_annotate"
        else "training"
    )
    return StreamingResponse(
        service.jobs.download_log(job_id),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{kind}-{job.id}.log"'},
    )


@router.get("/jobs/{job_id}/training-report")
def training_report(job_id: str, service: Service) -> TrainingReport | None:
    return service.training_reports.get(job_id)


@router.post("/jobs/{job_id}/training-report", status_code=202)
def create_training_report(job_id: str, service: Service) -> Job:
    return service.training_reports.start(job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, service: Service) -> Job:
    return service.jobs.cancel(job_id)


@router.get("/jobs/{job_id}/events")
def events(job_id: str, service: Service) -> StreamingResponse:
    service.store.get(Job, job_id)

    def stream() -> Iterator[str]:
        previous = ""
        while True:
            item = service.store.get(Job, job_id)
            data = item.model_dump_json()
            if data != previous:
                yield f"event: job\ndata: {data}\n\n"
                previous = data
            if item.status in TERMINAL_STATUSES:
                return
            time.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/regions/{region_id}")
def region(region_id: str, service: Service) -> RegionProposal:
    return service.store.get(RegionProposal, region_id)


@router.get("/projects/{project_id}/recipes")
def recipes(project_id: str, service: Service) -> list[RecipeInfo]:
    project = service.store.get(Project, project_id)
    return [
        recipe
        for recipe in service.models.recipes.list()
        if not recipe.demo_only or project.is_demo
    ]


@router.get("/projects/{project_id}/learners")
def learners(project_id: str, service: Service) -> list[Learner]:
    return [learner for learner in service.store.list(Learner, project_id) if not learner.archived]


@router.post("/projects/{project_id}/learners", status_code=201)
def create_learner(project_id: str, body: LearnerCreate, service: Service) -> Learner:
    return service.learning.create(project_id, body)


@router.post("/projects/{project_id}/learners/{learner_id}/train", status_code=202)
def train_learner(
    project_id: str, learner_id: str, body: StartTraining, service: Service, user: Principal
) -> Job:
    return service.learning.start(project_id, learner_id, body, authorized_by=user.id)


@router.patch("/projects/{project_id}/learners/{learner_id}")
def update_learner(
    project_id: str, learner_id: str, body: LearnerUpdate, service: Service
) -> Learner:
    return service.learning.update(project_id, learner_id, body)


@router.delete("/projects/{project_id}/learners/{learner_id}")
def delete_learner(
    project_id: str, learner_id: str, body: DeleteModelRequest, service: Service
) -> Learner:
    return service.learning.delete(project_id, learner_id, body)


@router.patch("/projects/{project_id}/models/{model_id}")
def update_model(
    project_id: str, model_id: str, body: ModelUpdate, service: Service
) -> ModelRecord:
    return service.models.update(project_id, model_id, body)


@router.delete("/projects/{project_id}/models/{model_id}")
def delete_model(
    project_id: str, model_id: str, body: DeleteModelRequest, service: Service
) -> ModelRecord:
    return service.models.delete(project_id, model_id, body)


@router.get("/projects/{project_id}/dataset-templates")
def dataset_templates(project_id: str, service: Service) -> list[DatasetTemplate]:
    service.store.get(Project, project_id)
    return service.dataset_templates.catalog()


@router.post("/projects/{project_id}/dataset-imports", status_code=202)
def import_dataset_template(
    project_id: str, body: DatasetTemplateImport, service: Service, user: Principal
) -> Job:
    return service.dataset_templates.start(project_id, body, user.id)


@router.post("/projects/{project_id}/review-decisions")
def review_decisions(
    project_id: str, body: BatchReviewRequest, service: Service, user: Principal
) -> list[ReviewDecision]:
    return service.reviews.decide_batch(project_id, body, user)
