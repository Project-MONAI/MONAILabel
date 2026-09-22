"""Annotation tools accept structured intent; geometry comes only from the viewer."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    AnnotateRequest,
    AssistantReply,
    BatchAnnotateRequest,
    BoxRequest,
    ClearSegmentsAction,
    Contract,
    ImageTiling,
    LabelColorsUpdate,
    RoiRequest,
)
from monailabel.core.tiling import image_tiles
from monailabel.server.labels import resolve_labels, update_colors

from .base import ToolContext, ToolRegistry
from .spatial import prompt_for


class AnnotationArgs(Contract):
    targets: list[str] = Field(
        default_factory=list,
        max_length=31,
        description=(
            "Anatomical/segmentation names; empty uses current labels. "
            "Correct obvious typos, e.g. nuclie to nuclei."
        ),
    )
    model_id: str | None = Field(
        default=None,
        description="Exact model ID. Prefer model_name for an explicitly named model.",
    )
    model_name: str | None = Field(
        default=None,
        description="Exact name from available models when the user names a model, e.g. 'SAM 2.1'. "
        "This overrides the viewer selection/default. Omit both name and ID to use selection.",
    )
    scope: Literal["auto", "current_slice", "selected_region", "full"] = "auto"
    tile_size: int | None = Field(
        default=None,
        ge=64,
        le=2048,
        description="Only for the full 2D image; never tile a selected region.",
    )


class BatchAnnotationArgs(Contract):
    limit: int = Field(
        default=5, ge=1, le=100, description="Number of unannotated images, in dataset order."
    )
    targets: list[str] = Field(default_factory=list, max_length=31)
    model_id: str | None = None
    model_name: str | None = Field(
        default=None, description="Exact available name of the explicitly requested model."
    )
    submit_for_review: bool = Field(
        default=False,
        description="True when asked to submit results for review; leaves reviews pending.",
    )


class RegionArgs(Contract):
    target: str = Field(min_length=1, max_length=80)
    model_id: str | None = None
    kind: Literal["box", "roi"] = "box"
    first_slice: int | None = Field(
        default=None, ge=1, description="ROI start, numbered from 1, inclusive."
    )
    last_slice: int | None = Field(
        default=None, ge=1, description="ROI end, numbered from 1, inclusive."
    )

    @model_validator(mode="after")
    def range_valid(self) -> "RegionArgs":
        if self.kind == "roi" and (
            self.first_slice is None
            or self.last_slice is None
            or self.first_slice > self.last_slice
        ):
            raise ValueError("Provide the inclusive first and last slice for a 3D ROI.")
        if self.kind == "box" and (self.first_slice is not None or self.last_slice is not None):
            raise ValueError("Slice ranges require an ROI.")
        return self


class RemoveRegionArgs(Contract):
    target: str = Field(min_length=1, max_length=80)
    all_matches: bool = False
    current_slice: bool = False


class ClearArgs(Contract):
    targets: list[str] = Field(
        default_factory=list,
        max_length=31,
        description="Exact foreground label names. For all annotations, omit targets and set "
        "all_targets=true; never use 'all' as a label.",
    )
    all_targets: bool = Field(
        default=False, description="True only for an explicit request to clear all segments."
    )
    scope: Literal["full", "current_slice", "selected_region"] = "full"


class ColorArgs(Contract):
    targets: list[str] = Field(min_length=1, max_length=31)
    color: Literal["anatomical"] | Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")] = Field(
        default="anatomical",
        description=(
            "Use 'anatomical' to restore the anatomical/default/standard color. "
            "Otherwise convert the requested named color to #RRGGBB."
        ),
    )


class ClassifyArgs(Contract):
    model_id: str | None = None
    categories: list[str] = Field(
        default_factory=lambda: ["Tumor", "Immune", "Stromal"], min_length=2, max_length=16
    )
    selected_only: bool = Field(
        default=False,
        description=(
            "False for 'classify the annotated nuclei'. True ONLY when the current request "
            "explicitly says selected nuclei/objects. An image ROI is not selected nuclei."
        ),
    )


def register(registry: ToolRegistry) -> None:
    ctx = registry.context
    registry.add(
        "annotate",
        (
            "Segment structures in the current sample with an annotation "
            "model. Full means all slices/whole image. Uses exact viewer "
            "crop for selected_region, no tiling. Current_slice needs a "
            "viewer slice."
        ),
        AnnotationArgs,
        lambda a: annotate(ctx, a),
        requires_asset=True,
        action="annotate",
    )
    registry.add(
        "annotate_batch",
        "Run segmentation on the first N unannotated images in dataset order, skipping "
        "evaluation-only data and existing annotations/proposals. Use for 'run segmentation "
        "for first 5 images and submit them for review', with submit_for_review=true. "
        "Runs full images/volumes without opening a viewer; needs no current sample. "
        "This job performs inference and optional submission together, never accepts reviews.",
        BatchAnnotationArgs,
        lambda a: annotate_batch(ctx, a),
        action="annotate",
    )
    registry.add(
        "locate_region",
        (
            "Create an editable bounding box on the current slice, or a "
            "3D ROI across an inclusive slice range using the selected "
            "vision model on EVERY slice. These are outlines, not segmentation "
            "masks."
        ),
        RegionArgs,
        lambda a: locate(ctx, a),
        requires_asset=True,
        action="annotate",
    )
    registry.add(
        "remove_regions",
        (
            "Remove named bounding boxes/ROIs from the current Slicer scene. "
            "Does not clear segmentation masks."
        ),
        RemoveRegionArgs,
        lambda a: remove(ctx, a),
        requires_asset=True,
        action="annotate",
    )
    registry.add(
        "clear_segments",
        (
            "Clear segmentation labels locally in a supported viewer, with undo. "
            "Choose named targets or explicitly all_targets. Full means the whole "
            "image/volume; current_slice requires a viewer slice; selected_region clears "
            "only inside the actual QuPath selection. Does not delete "
            "samples, label definitions, projects, or saved revisions."
        ),
        ClearArgs,
        lambda a: clear(ctx, a),
        requires_asset=True,
        action="edit",
    )
    registry.add(
        "set_label_color",
        (
            "Change the shared project display color of existing labels, "
            "without changing segmentation voxels. Use color='anatomical' to restore the "
            "Slicer Generic Anatomy palette. Does not invoke an annotation model."
        ),
        ColorArgs,
        lambda a: set_color(ctx, a),
        action="edit",
    )
    registry.add(
        "classify_objects",
        (
            "Classify annotated nuclei/objects in QuPath into cell types "
            "or user categories, using a vision model. Requests viewer "
            "object capture if needed."
        ),
        ClassifyArgs,
        lambda a: classify(ctx, a),
        requires_asset=True,
        action="annotate",
    )


def annotate(ctx: ToolContext, args: AnnotationArgs) -> AssistantReply:
    asset, project, context, service = ctx.asset, ctx.project, ctx.context, ctx.service
    targets = args.targets or [
        label.name
        for label in project.labels
        if label.id and (not context.label_ids or label.id in context.label_ids)
    ]
    model_id = ctx.model_id(args.model_id, args.model_name, targets=targets)
    scope = args.scope
    if scope == "auto":
        scope = (
            "selected_region"
            if context.image_region
            else "current_slice"
            if context.slice
            else "full"
        )
    region = context.image_region if scope == "selected_region" else None
    if scope == "selected_region" and region is None:
        raise DomainError(
            "Select one area in QuPath before annotating a region. The "
            "full image was not sent to a model."
        )
    if args.tile_size and (scope != "full" or asset.kind != "image2d"):
        raise DomainError(
            "Tile size applies to a full 2D image or pathology slide; selected "
            "regions are one crop."
        )
    if args.targets:
        names = list(dict.fromkeys(n.strip() for n in args.targets))
        project, labels = resolve_labels(service.store, project.id, names, model_id)
    else:
        labels = context.label_ids or [label.id for label in project.labels if label.id]
    if not labels:
        raise DomainError("Name a structure to annotate, for example spleen or liver.")
    model_ids = {
        model_id or project.defaults.get(label) or project.annotation_model_id for label in labels
    }
    if None in model_ids:
        return AssistantReply(
            assistant="model",
            message=(
                "No annotation model is selected for these labels. Open Models, "
                "connect a model, and set its label defaults."
            ),
            data={"form": "model"},
        )
    models = [service.models.get(project.id, str(identifier)) for identifier in model_ids]
    spatial = any(service.models.requires_spatial(m) for m in models)
    is_slice = scope == "current_slice"
    slice_volume = (
        scope == "full"
        and asset.kind == "volume3d"
        and (spatial or any(service.models.requires_2d(m) for m in models))
    )
    if (is_slice or slice_volume) and context.slice is None:
        raise DomainError(
            "Select a slice/view and intensity window in Slicer before annotating slices."
        )
    tiling = None
    if scope == "full" and asset.kind == "image2d":
        tiling = ImageTiling(tile_size=args.tile_size) if args.tile_size else context.image_tiling
    spatial_prompt = context.spatial_prompt
    if spatial and context.spatial_objects and spatial_prompt is None:
        if len(labels) != 1 or context.slice is None:
            raise DomainError("Choose one target and a source slice for SAM.")
        target = next(label.name for label in project.labels if label.id == labels[0])
        spatial_prompt = prompt_for(context.spatial_objects, context.slice, target)
    job = service.annotations.annotate(
        asset.id,
        AnnotateRequest(
            model_id=model_id,
            label_ids=labels,
            prompt=project.instructions + "\n" + ctx.message,
            slice=context.slice if is_slice or slice_volume else None,
            all_slices=slice_volume,
            image_region=region,
            image_tiling=tiling,
            spatial_prompt=spatial_prompt,
        ),
    )
    description = (
        f"the selected {region.width} × {region.height} pixel region as one crop, without tiling"
        if region
        else "the volume by propagating your spatial prompt in both directions"
        if slice_volume and spatial
        else f"all {asset.spatial_shape[context.slice.axis]} slices, one request per slice"
        if slice_volume and context.slice
        else "the selected slice"
        if is_slice
        else "the full image/volume"
    )
    if tiling:
        count = len(image_tiles(asset.spatial_shape[0], asset.spatial_shape[1], tiling))
        description += (
            f" in {count} native-resolution tiles, up to {tiling.tile_size} "
            f"× {tiling.tile_size} pixels"
        )
    return AssistantReply(
        assistant="annotation",
        job_id=job.id,
        message=(
            f"Annotating {description} using {', '.join(m.name for m in models)}. "
            f"You can cancel this job."
        ),
        data={"project": project.model_dump(mode="json")},
    )


def annotate_batch(ctx: ToolContext, args: BatchAnnotationArgs) -> AssistantReply:
    project, service = ctx.project, ctx.service
    if args.submit_for_review:
        service.auth.require(ctx.user, project.id, "edit")
    model_id = ctx.model_id(args.model_id, args.model_name)
    if args.targets:
        project, labels = resolve_labels(service.store, project.id, args.targets, model_id)
    else:
        labels = ctx.context.label_ids or [label.id for label in project.labels if label.id]
    if not labels:
        raise DomainError("Name a structure to annotate, for example spleen or liver.")
    job = service.batch_annotations.start(
        project.id,
        BatchAnnotateRequest(
            limit=args.limit,
            model_id=model_id,
            label_ids=labels,
            submit_for_review=args.submit_for_review,
            prompt=project.instructions + "\n" + ctx.message,
        ),
        ctx.user.id,
    )
    selected = job.request["asset_ids"]
    assert isinstance(selected, list)
    destination = (
        "submit the results to Reviews as pending"
        if args.submit_for_review
        else "save proposals for viewer review"
    )
    return AssistantReply(
        assistant="annotation",
        job_id=job.id,
        message=f"Segmenting the first {len(selected)} eligible images in dataset order. "
        f"I will {destination}. Evaluation-only cases and existing annotations are skipped. "
        "You can follow progress or cancel in Activity.",
        data={"project": project.model_dump(mode="json")},
    )


def locate(ctx: ToolContext, args: RegionArgs) -> AssistantReply:
    asset, context, model = ctx.asset, ctx.context, ctx.model(args.model_id)
    if asset.kind != "volume3d" or context.slice is None:
        raise DomainError("Open the volume in Slicer or OHIF to send its slice context.")
    request: BoxRequest
    if args.kind == "roi":
        if "roi" not in context.viewer_actions:
            raise DomainError("Open this sample in an updated Slicer assistant for 3D ROIs.")
        assert args.first_slice is not None and args.last_slice is not None
        if args.last_slice > asset.spatial_shape[context.slice.axis]:
            raise DomainError("ROI range exceeds the number of slices in the selected view.")
        request = RoiRequest(
            model_id=model.id,
            target=args.target,
            prompt=ctx.message,
            slice=context.slice.model_copy(update={"index": args.first_slice - 1}),
            end_index=args.last_slice - 1,
        )
        message = (
            f"Locating {args.target} with {model.name} "
            f"on each of {args.last_slice - args.first_slice + 1} "
            f"slices ({args.first_slice}–{args.last_slice}, numbered from "
            f"1, inclusive). Their combined bounds form an editable 3D ROI."
        )
    else:
        request = BoxRequest(
            model_id=model.id, target=args.target, prompt=ctx.message, slice=context.slice
        )
        message = (
            f"Locating a box for {args.target} on the selected slice using "
            f"{model.name}. This creates an editable viewer outline, separate "
            f"from segmentation masks."
        )
    job = ctx.service.regions.locate(asset.id, request)
    return AssistantReply(assistant="annotation", job_id=job.id, message=message)


def remove(ctx: ToolContext, args: RemoveRegionArgs) -> AssistantReply:
    asset = ctx.asset
    if "remove_regions" not in ctx.context.viewer_actions:
        return AssistantReply(
            assistant="viewer",
            message=(
                "Run this in an updated Slicer assistant. The box lives in "
                "the viewer; update the bridge in place to keep your current "
                "scene and edits."
            ),
        )
    if args.current_slice and ctx.context.slice is None:
        raise DomainError("Select a source slice before removing a box from that slice.")
    return AssistantReply(
        assistant="viewer",
        message=f"Removing the {args.target} box in the current viewer.",
        data={
            "client_action": "remove_regions",
            "project_id": ctx.project_id,
            "asset_id": asset.id,
            "target": args.target,
            "all_matches": args.all_matches,
            "slice": ctx.context.slice.model_dump(mode="json")
            if args.current_slice and ctx.context.slice
            else None,
        },
    )


def clear(ctx: ToolContext, args: ClearArgs) -> AssistantReply:
    asset, context, project = ctx.asset, ctx.context, ctx.project
    if "clear_segments" not in context.viewer_actions:
        return AssistantReply(
            assistant="viewer",
            message=(
                "This viewer session does not support clearing segments through chat. "
                "Use its manual editing controls, or save your work and reopen the "
                "sample in an updated viewer. "
                "Saved annotation revisions are unchanged."
            ),
        )
    if args.all_targets and args.targets or not args.all_targets and not args.targets:
        raise DomainError("Choose named labels or explicitly all segments to clear.")
    labels = [label for label in project.labels if label.id]
    if not args.all_targets:
        names = {name.casefold() for name in args.targets}
        if not names <= {label.name.casefold() for label in labels}:
            raise DomainError("Unknown label to clear. Choose an existing segmentation label.")
        labels = [label for label in labels if label.name.casefold() in names]
    if not labels:
        raise DomainError("There are no foreground labels to clear.")
    region = context.image_region if args.scope == "selected_region" else None
    if args.scope == "selected_region" and region is None:
        raise DomainError("Select a supported image region before clearing labels inside it.")
    if region and (
        asset.kind != "image2d"
        or region.x + region.width > asset.spatial_shape[1]
        or region.y + region.height > asset.spatial_shape[0]
    ):
        raise DomainError("Select a region inside the source 2D image.")
    slice_scope = context.slice if args.scope == "current_slice" else None
    if args.scope == "current_slice" and (
        asset.kind != "volume3d"
        or slice_scope is None
        or slice_scope.index >= asset.spatial_shape[slice_scope.axis]
    ):
        raise DomainError("Select a valid source-volume slice before clearing labels on it.")
    label_names = ", ".join(label.name for label in labels)
    scope = (
        "on the requested slice"
        if slice_scope
        else "inside the selected region"
        if region
        else "throughout the current volume"
        if asset.kind == "volume3d"
        else "in the current image"
    )
    return AssistantReply(
        assistant="viewer",
        message=(f"Clearing {label_names} {scope}. This is an undoable local edit."),
        data=ClearSegmentsAction(
            project_id=project.id,
            asset_id=asset.id,
            base_revision=asset.revision,
            label_ids=[label.id for label in labels],
            image_region=region,
            slice=slice_scope,
        ).model_dump(mode="json"),
    )


def set_color(ctx: ToolContext, args: ColorArgs) -> AssistantReply:
    project = update_colors(
        ctx.service.store,
        ctx.project.id,
        LabelColorsUpdate(
            targets=args.targets,
            color=None if args.color == "anatomical" else args.color,
            base_version=ctx.project.version,
        ),
    )
    return AssistantReply(
        assistant="annotation",
        message=(
            f"Updated the project color for {', '.join(args.targets)}. "
            "Segmentation voxels are unchanged."
        ),
        data={"project": project.model_dump(mode="json")},
    )


def classify(ctx: ToolContext, args: ClassifyArgs) -> AssistantReply:
    if "classify_objects" not in ctx.context.viewer_actions:
        return AssistantReply(
            assistant="annotation",
            message=(
                "Object classification needs the updated QuPath assistant. "
                "Save your draft and reopen the sample."
            ),
        )
    asset = ctx.asset
    if ctx.context.classification is None:
        return AssistantReply(
            assistant="annotation",
            message=(
                "Preparing the annotated objects for classification. Categories "
                "can be edited in QuPath."
            ),
            data={
                "client_action": "configure_classification",
                "categories": list(args.categories),
                "selected_only": args.selected_only,
                "model_id": args.model_id,
            },
        )
    classification = ctx.context.classification.model_copy(
        update={
            "model_id": args.model_id
            or ctx.context.model_id
            or ctx.context.classification.model_id,
            "prompt": ctx.message,
        }
    )
    job = ctx.service.classifications.classify(asset.id, classification)
    return AssistantReply(
        assistant="annotation",
        job_id=job.id,
        message=(
            f"Classifying {len(classification.objects)} annotated objects "
            f"into {', '.join(classification.categories)}. Uncertain objects "
            f"may be left unclassified. This creates editable candidate "
            f"classifications; you can cancel the job."
        ),
    )
