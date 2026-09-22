"""Typed video actions, sharing the web workspace's revision and role checks."""

from typing import Literal

from pydantic import Field

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import AssistantReply, Contract, DecisionRequest, VideoPrompt
from monailabel.core.video import (
    ClearVideoAction,
    PolygonKeyframe,
    TrackKeyframe,
    VideoAsset,
    VideoDecision,
    VideoEditorRequest,
    VideoFindTrackingRequest,
    VideoTrackingRequest,
)
from monailabel.server.video.models import VideoEditor

from .base import Empty, ToolContext, ToolRegistry


class TrackVideo(Contract):
    frame_count: int = Field(default=16, ge=1, le=200_000)
    scope: Literal["frames", "current_frame", "whole_video"] = "frames"
    output: Literal["box", "polygon"] | None = Field(
        default=None,
        description="Use box for locate/find/bounding box and polygon for segment/outline "
        "in the current request. Do not inherit a shape from an earlier request. "
        "Omit only to preserve a selected track's shape.",
    )


class FindVideoTool(TrackVideo):
    model_id: str | None = Field(
        default=None,
        description="Exact registered model ID. Omit to use the selection/default or model_name.",
    )
    model_name: str | None = Field(
        default=None,
        description="Exact name from available models, only when the user names a model. "
        "Otherwise omit both model_name and model_id to use the selection or default. "
        "Never put a model ID or the tracker name here.",
    )
    label_name: str | None = None
    prompt: str = Field(default="", max_length=2000)


class OpenVideo(VideoEditorRequest):
    video_id: str


class SubmitVideo(Contract):
    video_id: str
    editor_id: str


class DecideVideo(VideoDecision):
    video_id: str


class ClearVideo(Contract):
    targets: list[str] = Field(default_factory=list, max_length=31)
    all_targets: bool = False
    selected_track: bool = False
    scope: Literal["current_frame", "frames", "whole_video"] = "current_frame"
    frame_count: int = Field(default=1, ge=1, le=200_000)


def draft_context(ctx: ToolContext) -> tuple[VideoPrompt, VideoAsset, VideoEditor]:
    prompt = ctx.context.video
    if prompt is None:
        raise DomainError("Open the video in CVAT and send this request from its assistant.")
    video = asset(ctx, prompt.video_id)
    editor = ctx.service.store.get(VideoEditor, prompt.editor_id)
    if editor.project_id != video.project_id or editor.asset_id != video.id or not editor.ready:
        raise DomainError("Choose a ready CVAT editor for this video.")
    ctx.service.auth.require(
        ctx.user, video.project_id, "review" if editor.mode == "review" else "annotate"
    )
    if (
        ctx.context.base_revision != video.revision
        or editor.base_revision != video.revision
        or editor.submitted_annotation_id
    ):
        raise Conflict("The video revision changed. Reopen it; your CVAT draft is preserved.")
    if prompt.frame >= video.frames:
        raise DomainError("Choose a source frame within the clip.")
    return prompt, video, editor


def asset(ctx: ToolContext, identifier: str) -> VideoAsset:
    video = ctx.service.store.get(VideoAsset, identifier)
    if video.project_id != ctx.project.id:
        raise DomainError("Choose a video in the current project.")
    return video


def register(registry: ToolRegistry) -> None:
    ctx = registry.context
    registry.add(
        "clear_video_annotations",
        "Clear local CVAT boxes/polygons, with undo. Choose named targets, explicitly "
        "all_targets for all labels, or selected_track for only the selected tool. "
        "scope=current_frame clears this frame; frames clears frame_count frames including "
        "this frame; whole_video clears from frame zero to the end. Preserve other frames, "
        "labels and saved revisions. Requires actual CVAT viewer context; no model is used.",
        ClearVideo,
        lambda args: clear(ctx, args),
        action="edit",
    )
    registry.add(
        "find_and_track_video_tool",
        "Locate one tool on the current CVAT source frame with the explicitly named or selected "
        "annotation model, or automatic default (GPT-6 Astra), then track with SAM 2.1. "
        "Use output=box for locate/bounding box, "
        "output=polygon for segment/outline. scope=current_frame annotates just the current frame; "
        "scope=frames uses frame_count including the current frame; scope=whole_video starts at "
        "frame zero and covers the clip. No drawn box required. "
        "Returns annotations that CVAT adds to the editable draft; never saves or submits it. "
        "Omit model_name and model_id to use the selection or automatic default. "
        "Pass an exact available model_name only when the user names a model. "
        "label_name must match "
        "a project label, or omit to use the selected label. prompt may describe which instance "
        "to find. Uses actual viewer frame and draft signature; never supply invented coordinates.",
        FindVideoTool,
        lambda args: find_and_track(ctx, args),
        action="edit",
    )
    registry.add(
        "track_selected_video_tool",
        "Use SAM 2.1 to track the visible rectangle or polygon selected in the CVAT viewer, "
        "from its current frame for frame_count frames including the seed. scope=current_frame "
        "uses one frame; whole_video requires a selected seed on frame zero. output=polygon "
        "segments from a selected box and adds a new polygon track; omit to preserve shape. "
        "Returns annotations that CVAT adds to the editable draft after checking for edits. "
        "Requires video context with the actual selected box, track ID and draft signature. "
        "Without a selection, uses the selected or default annotation model to find a tool. "
        "From workspace chat, opens "
        "CVAT automatically only when the project has exactly one video.",
        TrackVideo,
        lambda args: track_selected(ctx, args),
        action="edit",
    )
    registry.add(
        "list_videos",
        "List imported video clips and their current revisions. Video annotation uses CVAT "
        "rectangle/polygon tracks and SAM 2 tracking. Accepted polygon frame ranges "
        "can train segmentation models; temporal tracking metrics are unavailable. "
        "To import clips, use open_form(video-import) in the workspace.",
        Empty,
        lambda _: AssistantReply(
            assistant="dataset",
            message="Video clips in this project.",
            data={
                "videos": [
                    v.model_dump(mode="json")
                    for v in ctx.service.store.list(VideoAsset, ctx.project.id)
                ]
            },
        ),
    )
    registry.add(
        "open_video_editor",
        "Open or resume a video's saved CVAT task at the observed base_revision. "
        "mode=review inspects submitted tracks in a separate task. Existing drafts are kept.",
        OpenVideo,
        lambda args: open_editor(ctx, args),
        action="read",
    )
    registry.add(
        "submit_video_tracks",
        "Submit tracks already saved in CVAT for review. Requires the exact editor_id "
        "returned by the editor job. Does not save unsaved CVAT browser edits.",
        SubmitVideo,
        lambda args: submit(ctx, args),
        action="edit",
    )
    registry.add(
        "review_video_tracks",
        "Accept or request changes to the exact submitted video base_revision. "
        "This decision does not include unsubmitted CVAT drafts.",
        DecideVideo,
        lambda args: decide(ctx, args),
        action="review",
    )


def clear(ctx: ToolContext, args: ClearVideo) -> AssistantReply:
    prompt, video, editor = draft_context(ctx)
    if "clear_video_annotations" not in ctx.context.viewer_actions:
        raise DomainError("Reopen CVAT to enable clearing annotations through its assistant.")
    if sum((bool(args.targets), args.all_targets, args.selected_track)) != 1:
        raise DomainError("Choose named labels, all annotations, or the selected track to clear.")
    labels = [label for label in ctx.project.labels if label.id in editor.label_map and label.id]
    if args.targets:
        names = {name.strip().casefold() for name in args.targets}
        if not names <= {label.name.casefold() for label in labels}:
            raise DomainError("Choose an existing tool label in this CVAT task.")
        labels = [label for label in labels if label.name.casefold() in names]
    if args.selected_track:
        if prompt.client_id is None or prompt.label_id is None:
            raise DomainError("Select a visible, unlocked rectangle or polygon track in CVAT.")
        labels = [label for label in labels if label.id == prompt.label_id]
    if not labels:
        raise DomainError("Choose a foreground tool label to clear.")
    start = 0 if args.scope == "whole_video" else prompt.frame
    stop = (
        video.frames
        if args.scope == "whole_video"
        else start + (args.frame_count if args.scope == "frames" else 1)
    )
    if stop > video.frames:
        raise DomainError("Choose a frame range within the clip.")
    return AssistantReply(
        assistant="viewer",
        message="Clearing the requested annotations in your CVAT draft.",
        data=ClearVideoAction(
            project_id=video.project_id,
            video_id=video.id,
            editor_id=editor.id,
            base_revision=video.revision,
            draft_signature=prompt.draft_signature,
            label_ids=[label.id for label in labels],
            client_id=prompt.client_id if args.selected_track else None,
            start=start,
            stop=stop,
        ).model_dump(mode="json"),
    )


def open_editor(ctx: ToolContext, args: OpenVideo) -> AssistantReply:
    video = asset(ctx, args.video_id)
    ctx.service.auth.require(
        ctx.user, video.project_id, "review" if args.mode == "review" else "annotate"
    )
    job = ctx.service.video_editors.start(
        video.id, VideoEditorRequest(base_revision=args.base_revision, mode=args.mode)
    )
    return AssistantReply(
        assistant="annotation", message="Preparing the CVAT editor.", job_id=job.id
    )


def submit(ctx: ToolContext, args: SubmitVideo) -> AssistantReply:
    video = asset(ctx, args.video_id)
    editor = ctx.service.store.get(VideoEditor, args.editor_id)
    ctx.service.auth.require(
        ctx.user, video.project_id, "review" if editor.mode == "review" else "annotate"
    )
    annotation = ctx.service.video_editors.submit(video.id, args.editor_id, ctx.user)
    return AssistantReply(
        assistant="annotation",
        message="Saved CVAT tracks submitted for review.",
        data={"video_id": video.id, "revision": annotation.revision},
    )


def decide(ctx: ToolContext, args: DecideVideo) -> AssistantReply:
    video = asset(ctx, args.video_id)
    decision = ctx.service.videos.decide(
        video.id,
        args.base_revision,
        DecisionRequest(verdict=args.verdict, comment=args.comment),
        ctx.user,
    )
    return AssistantReply(
        assistant="review",
        message="Video review decision saved.",
        data={"decision_id": decision.id, "revision": decision.revision},
    )


def track_selected(ctx: ToolContext, args: TrackVideo) -> AssistantReply:
    video = ctx.context.video
    instructions = (
        "In CVAT, go to a frame where the tool is visible. Choose the rectangle drawing tool, "
        "choose Track, and draw a box around the tool. If you already drew a rectangle track, "
        "select it on that frame and make sure it is unlocked and visible. "
        f'Then ask "Track this tool for {args.frame_count} frames" in the CVAT Assistant. '
        "Alternatively, ask to locate or segment the tool using the default annotation model. "
        "Tracking has not started yet; SAM 2 needs a starting annotation."
    )
    if video is None:
        videos = ctx.service.store.list(VideoAsset, ctx.project.id)
        if len(videos) == 1:
            selected = videos[0]
            opened = open_editor(
                ctx, OpenVideo(video_id=selected.id, base_revision=selected.revision)
            )
            return opened.model_copy(
                update={
                    "message": "Preparing CVAT; the editor will open automatically when ready. "
                    + instructions
                }
            )
        return AssistantReply(
            assistant="annotation",
            message=(
                "Choose the clip you want in Datasets and click its CVAT button. "
                if videos
                else "Import a video in Datasets, then click its CVAT button. "
            )
            + instructions,
        )
    if (
        (video.box is None and video.points is None)
        or video.client_id is None
        or video.label_id is None
        or ctx.context.base_revision is None
    ):
        if ctx.context.base_revision is not None:
            return find_and_track(
                ctx,
                FindVideoTool(frame_count=args.frame_count, scope=args.scope, output=args.output),
            )
        return AssistantReply(assistant="annotation", message=instructions)
    selected = asset(ctx, video.video_id)
    editor = ctx.service.store.get(VideoEditor, video.editor_id)
    ctx.service.auth.require(
        ctx.user, selected.project_id, "review" if editor.mode == "review" else "annotate"
    )
    if args.scope == "whole_video" and video.frame != 0:
        raise DomainError(
            "Select the tool on frame 0 to track the whole video, "
            "or ask to find it with an annotation model."
        )
    seed = (
        PolygonKeyframe(frame=video.frame, points=video.points, occluded=video.occluded)
        if video.points
        else TrackKeyframe(frame=video.frame, box=video.box or [], occluded=video.occluded)
    )
    request = VideoTrackingRequest(
        editor_id=video.editor_id,
        base_revision=ctx.context.base_revision,
        client_id=video.client_id,
        label_id=video.label_id,
        seed=seed,
        output=args.output or ("polygon" if video.points else "box"),
        frame_count=selected.frames
        if args.scope == "whole_video"
        else 1
        if args.scope == "current_frame"
        else args.frame_count,
        draft_signature=video.draft_signature,
    )
    job = ctx.service.video_tracking.start(selected.id, request)
    return AssistantReply(
        assistant="annotation",
        message=(
            "Preparing the selected annotation. "
            if request.frame_count == 1
            and isinstance(seed, PolygonKeyframe) == (request.output == "polygon")
            else "Tracking the selected tool with SAM 2.1. "
        )
        + "The result will appear in your CVAT draft for inspection and correction.",
        job_id=job.id,
    )


def find_and_track(ctx: ToolContext, args: FindVideoTool) -> AssistantReply:
    video = ctx.context.video
    if video is None:
        return track_selected(
            ctx, TrackVideo(frame_count=args.frame_count, scope=args.scope, output=args.output)
        )
    if ctx.context.base_revision is None:
        raise DomainError("Reload the CVAT panel to obtain the current video revision.")
    labels = [label for label in ctx.project.labels if label.id]
    if args.label_name:
        labels = [
            label for label in labels if label.name.casefold() == args.label_name.strip().casefold()
        ]
    elif ctx.context.label_ids:
        labels = [label for label in labels if label.id in (ctx.context.label_ids or [])]
    if len(labels) != 1:
        raise DomainError(
            "Name one project tool label in your prompt, or select its track in CVAT."
        )
    selected = asset(ctx, video.video_id)
    identifier = ctx.model_id(
        args.model_id, args.model_name, targets=[labels[0].name], source=selected
    )
    if not identifier:
        raise DomainError("Connect an annotation model in Models to locate or segment the tool.")
    editor = ctx.service.store.get(VideoEditor, video.editor_id)
    ctx.service.auth.require(
        ctx.user, selected.project_id, "review" if editor.mode == "review" else "annotate"
    )
    job = ctx.service.video_tracking.find(
        selected.id,
        VideoFindTrackingRequest(
            editor_id=video.editor_id,
            base_revision=ctx.context.base_revision,
            model_id=identifier,
            label_id=labels[0].id,
            prompt=args.prompt,
            output=args.output or "box",
            frame=0 if args.scope == "whole_video" else video.frame,
            frame_count=selected.frames
            if args.scope == "whole_video"
            else 1
            if args.scope == "current_frame"
            else args.frame_count,
            draft_signature=video.draft_signature,
        ),
    )
    model = ctx.service.models.get(selected.project_id, identifier)
    count = (
        selected.frames
        if args.scope == "whole_video"
        else 1
        if args.scope == "current_frame"
        else args.frame_count
    )
    action = "Segmenting" if args.output == "polygon" else "Locating"
    following = f", then tracking {count} frames with SAM 2.1" if count > 1 else " on this frame"
    return AssistantReply(
        assistant="annotation",
        job_id=job.id,
        message=f"{action} {labels[0].name} with {model.name}{following}. "
        "The result will appear in your CVAT draft for inspection and correction.",
    )
