"""Authenticated clip import, CVAT launch, immutable tracks and review."""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import Field

from monailabel.core.errors import DomainError
from monailabel.core.models import Contract, Job, Project, ReviewDecision
from monailabel.core.video import (
    TrackAnnotation,
    TrackSubmission,
    VideoAsset,
    VideoDecision,
    VideoEditorRequest,
    VideoFindTrackingRequest,
    VideoImport,
    VideoMetadata,
    VideoTrackingProposal,
    VideoTrackingRequest,
)
from monailabel.server.access import Principal, Service, authorize
from monailabel.server.video_editor import VideoEditor
from monailabel.server.videos import MAX_VIDEO_BYTES

router = APIRouter(prefix="/api", dependencies=[Depends(authorize)])


@router.get("/projects/{project_id}/videos")
def videos(project_id: str, service: Service) -> list[VideoAsset]:
    service.store.get(Project, project_id)
    return service.store.list(VideoAsset, project_id)


@router.get("/projects/{project_id}/video-capabilities")
def capabilities(project_id: str, service: Service) -> dict[str, bool]:
    return {
        "cvat": service.video_editors.managed or service.video_editors.client is not None,
        "managed": service.video_editors.managed,
    }


@router.post("/projects/{project_id}/videos/upload", status_code=201)
async def upload(
    project_id: str,
    request: Request,
    metadata: Annotated[VideoImport, Query()],
    service: Service,
) -> VideoAsset:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError as exc:
            raise DomainError("Invalid Content-Length.", status=400) from exc
        if size < 0:
            raise DomainError("Invalid Content-Length.", status=400)
        if size > MAX_VIDEO_BYTES:
            raise DomainError("Video exceeds 2 GiB.", status=413)
    with tempfile.TemporaryDirectory(prefix="monailabel-video-") as directory:
        path = Path(directory) / "clip.bin"
        with path.open("wb") as stream:
            size = 0
            async for block in request.stream():
                size += len(block)
                if size > MAX_VIDEO_BYTES:
                    raise DomainError("Video exceeds 2 GiB.", status=413)
                await run_in_threadpool(stream.write, block)
        return await run_in_threadpool(service.videos.import_file, project_id, metadata, path)


@router.get("/videos/{video_id}/metadata")
def metadata(video_id: str, service: Service) -> VideoMetadata:
    asset = service.store.get(VideoAsset, video_id)
    return VideoMetadata.model_validate_json(service.artifacts.read(asset.metadata_key))


@router.get("/videos/{video_id}/source")
def source(video_id: str, service: Service) -> FileResponse:
    asset = service.store.get(VideoAsset, video_id)
    return FileResponse(service.artifacts.path(asset.source_key), filename=asset.name)


@router.get("/videos/{video_id}/tracks")
def tracks(video_id: str, service: Service) -> TrackSubmission:
    asset = service.store.get(VideoAsset, video_id)
    return TrackSubmission(base_revision=asset.revision, document=service.videos.document(asset))


@router.get("/videos/{video_id}/revisions")
def revisions(video_id: str, service: Service) -> list[TrackAnnotation]:
    asset = service.store.get(VideoAsset, video_id)
    return [
        item
        for item in service.store.list(TrackAnnotation, asset.project_id)
        if item.asset_id == video_id
    ]


@router.post("/videos/{video_id}/review", status_code=201)
def submit_tracks(
    video_id: str, body: TrackSubmission, service: Service, user: Principal
) -> TrackAnnotation:
    return service.videos.submit(video_id, body, user)


@router.post("/videos/{video_id}/decision")
def decide(video_id: str, body: VideoDecision, service: Service, user: Principal) -> ReviewDecision:
    from monailabel.core.models import DecisionRequest

    return service.videos.decide(
        video_id,
        body.base_revision,
        DecisionRequest(verdict=body.verdict, comment=body.comment),
        user,
    )


@router.post("/videos/{video_id}/editor", status_code=202)
def editor(video_id: str, body: VideoEditorRequest, service: Service, user: Principal) -> Job:
    asset = service.store.get(VideoAsset, video_id)
    service.auth.require(user, asset.project_id, "review" if body.mode == "review" else "annotate")
    return service.video_editors.start(video_id, body)


@router.get("/videos/{video_id}/editors")
def editors(video_id: str, service: Service) -> list[VideoEditor]:
    asset = service.store.get(VideoAsset, video_id)
    return [
        item
        for item in service.store.list(VideoEditor, asset.project_id)
        if item.asset_id == video_id and item.ready
    ]


class EditorSubmission(Contract):
    editor_id: str = Field(min_length=1)


@router.post("/videos/{video_id}/cvat-submit", status_code=201)
def submit_editor(
    video_id: str, body: EditorSubmission, service: Service, user: Principal
) -> TrackAnnotation:
    editor = service.store.get(VideoEditor, body.editor_id)
    service.auth.require(
        user, editor.project_id, "review" if editor.mode == "review" else "annotate"
    )
    return service.video_editors.submit(video_id, body.editor_id, user)


@router.post("/videos/{video_id}/track", status_code=202)
def track_video(
    video_id: str, body: VideoTrackingRequest, service: Service, user: Principal
) -> Job:
    asset = service.store.get(VideoAsset, video_id)
    editor = service.store.get(VideoEditor, body.editor_id)
    service.auth.require(
        user, asset.project_id, "review" if editor.mode == "review" else "annotate"
    )
    return service.video_tracking.start(video_id, body)


@router.post("/videos/{video_id}/find-and-track", status_code=202)
def find_and_track_video(
    video_id: str, body: VideoFindTrackingRequest, service: Service, user: Principal
) -> Job:
    asset = service.store.get(VideoAsset, video_id)
    editor = service.store.get(VideoEditor, body.editor_id)
    service.auth.require(
        user, asset.project_id, "review" if editor.mode == "review" else "annotate"
    )
    return service.video_tracking.find(video_id, body)


@router.get("/videos/{video_id}/tracking-proposals/{proposal_id}")
def tracking_proposal(video_id: str, proposal_id: str, service: Service) -> VideoTrackingProposal:
    proposal = service.store.get(VideoTrackingProposal, proposal_id)
    if proposal.asset_id != video_id:
        raise DomainError("This proposal belongs to another video.", status=404)
    service.video_tracking.validate(video_id, proposal.request)
    return proposal


@router.get("/videos/{video_id}/tracking-proposals/{proposal_id}/masks")
def proposal_masks(video_id: str, proposal_id: str, service: Service) -> FileResponse:
    proposal = service.store.get(VideoTrackingProposal, proposal_id)
    if proposal.asset_id != video_id or not proposal.masks_key:
        raise DomainError("No source masks are available for this proposal.", status=404)
    return FileResponse(
        service.artifacts.path(proposal.masks_key),
        media_type="application/zip",
        filename="source-frame-masks.zip",
    )
