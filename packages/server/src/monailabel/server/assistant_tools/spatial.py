# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed local spatial edits; the coordinator supplies intent, the viewer supplies geometry."""

import re
from collections.abc import Container
from typing import Annotated, Literal

from pydantic import Field

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    AssistantReply,
    Contract,
    PromptPoint,
    SliceScope,
    SpatialEditAction,
    SpatialObject,
    SpatialPrompt,
    new_id,
)

from .base import ToolContext, ToolRegistry

Coordinate = Annotated[float, Field(allow_inf_nan=False, ge=0)]


class SpatialArgs(Contract):
    operation: Literal["add", "move", "clear"] = Field(
        description="add creates a NEW hint. move replaces an EXISTING hint's coordinates: "
        "use move for move, resize, adjust, or change box bounds. clear removes matching hints."
    )
    kind: Literal["point", "box", "all"] = "point"
    target: str | None = Field(default=None, min_length=1, max_length=80)
    all_targets: bool = Field(
        default=False,
        description="True when clearing plural hints without a named label, e.g. clear negative "
        "points on this slice. False when the request names an organ/target or selected hint. "
        "Whole volume changes scope, never this flag. When true, OMIT target.",
    )
    polarity: Literal["positive", "negative", "all"] = Field(
        description="REQUIRED explicit point filter: negative for exclusion points, positive "
        "for inclusion points, all only for both polarities or box-only operations."
    )
    object_id: str | None = Field(default=None, description="Exact viewer object ID when needed.")
    scope: Literal["current_slice", "full"] = "current_slice"
    coordinates: list[Coordinate] | list[list[Coordinate]] | None = Field(
        default=None,
        min_length=1,
        max_length=3,
        description=(
            "User-supplied zero-based source coordinates: [[x,y]] for point or "
            "[[x1,y1],[x2,y2]] for box. In-plane axes are the two source axes other than "
            "slice.axis. Copy literal numbers from the user's request ONLY. "
            "For a point in a box OMIT coordinates and set box_center=true; never calculate it."
        ),
    )
    box_center: bool = Field(
        default=False,
        description=(
            "Explicit request for an editable point at an existing box center; "
            "not anatomical localization."
        ),
    )


def on_slice(item: SpatialObject, scope: SliceScope | None) -> bool:
    if scope is None:
        return all(len(p) == 2 for p in item.coordinates)
    return all(len(p) == 3 and abs(p[scope.axis] - scope.index) <= 0.5 for p in item.coordinates)


def choose(items: list[SpatialObject]) -> SpatialObject:
    selected = [item for item in items if item.selected]
    candidates = selected or items
    if len(candidates) != 1:
        raise DomainError(
            "Select exactly one matching hint in the viewer, or identify its object ID."
        )
    return candidates[0]


def prompt_for(
    items: list[SpatialObject],
    scope: SliceScope | None,
    target: str,
    *,
    full_volume: bool = False,
    inputs: Container[str] | None = None,
) -> SpatialPrompt:
    candidates = [
        i
        for i in items
        if (full_volume or on_slice(i, scope))
        and (not i.target or i.target.casefold() == target.casefold())
        and (
            inputs is None
            or ("box" if i.kind == "box" else "positive_point" if i.positive else "negative_point")
            in inputs
        )
    ]
    boxes = [i for i in candidates if i.kind == "box"]
    box = choose(boxes) if boxes else None
    if any(v < 0 for item in candidates for point in item.coordinates for v in point):
        raise DomainError("Move or clear hints outside the source image before running inference.")
    points = [
        PromptPoint(coordinates=i.coordinates[0], positive=i.positive)
        for i in candidates
        if i.kind == "point"
    ]
    if not box and not any(p.positive for p in points):
        raise DomainError(
            "Add a box or positive point for this target before running the spatial model."
        )
    return SpatialPrompt(box=box.coordinates if box else None, points=points)


def register(registry: ToolRegistry) -> None:
    registry.add(
        "edit_spatial_prompts",
        "Create, move/resize, or clear editable spatial points/boxes in Slicer, OHIF or QuPath "
        "for nnInteractive, MedSAM2 or SAM. "
        "Never edits segmentation masks. Add/move needs explicit user voxel coordinates, "
        "or box_center=true for an explicitly requested point in a box. "
        "To let the user draw a box or click a point without coordinates, "
        "activate set_interaction_mode instead. "
        "Organ localization without coordinates uses locate_region with an explicitly "
        "chosen capable model. "
        "Clear uses target or explicit all_targets; kind=all clears both boxes and points. "
        "Whole volume changes scope only. Clear spleen boxes in the whole volume: "
        "target=Spleen, kind=box, scope=full, polarity=all. "
        "Clear all spatial hints means scope=full, kind=all, polarity=all, all_targets=true. "
        "Missing coordinates/ambiguous selection require a question; never fabricate anatomy.",
        SpatialArgs,
        lambda args: edit(registry.context, args),
        requires_asset=True,
        action="edit",
    )


def edit(ctx: ToolContext, args: SpatialArgs) -> AssistantReply:
    asset, context = ctx.asset, ctx.context
    if "edit_spatial_prompts" not in context.viewer_actions:
        raise DomainError(
            "Open an updated Slicer, OHIF or QuPath session to edit spatial prompts through chat."
        )
    scope = context.slice
    if asset.kind == "volume3d" and (
        scope is None or scope.index >= asset.spatial_shape[scope.axis]
    ):
        raise DomainError("Select a source-aligned volume slice before editing spatial prompts.")
    if context.base_revision is None:
        raise DomainError("Refresh the viewer to supply the current annotation revision.")
    items = context.spatial_objects
    if any(len(point) != len(asset.spatial_shape) for item in items for point in item.coordinates):
        raise DomainError("Viewer input dimensions do not match the source image.")
    if len({i.id for i in items}) != len(items):
        raise DomainError("Viewer hint IDs must be unique.")
    if args.all_targets and args.target:
        raise DomainError(
            "Choose a target or all targets, not both. Omit target when all_targets=true.",
            code="invalid_tool_arguments",
        )
    matches = [
        i
        for i in items
        if (args.scope == "full" or on_slice(i, scope))
        and (args.target is None or i.target.casefold() == args.target.casefold())
        and (args.kind == "all" or i.kind == args.kind)
        and (
            i.kind != "point"
            or args.polarity == "all"
            or i.positive == (args.polarity == "positive")
        )
        and (args.object_id is None or i.id == args.object_id)
    ]
    upsert: list[SpatialObject] = []
    remove: list[str] = []
    if args.operation == "clear":
        if args.coordinates or args.box_center:
            raise DomainError("Clearing hints does not accept replacement coordinates.")
        if not args.target and not args.all_targets and not args.object_id:
            # A uniquely selected hint is an explicit scope; never silently clear everything.
            matches = [choose([i for i in matches if i.selected])]
        remove = [i.id for i in matches]
    else:
        if args.kind == "all" or args.scope != "current_slice" or args.all_targets:
            raise DomainError("Add/move one point or box on the current slice.")
        if args.operation == "add" and args.kind == "point" and args.polarity == "all":
            raise DomainError("Choose positive or negative for the new point.")
        old = choose(matches) if args.operation == "move" else None
        coords: list[list[float]] | None = None
        if args.coordinates:
            coords = (
                [args.coordinates]
                if isinstance(args.coordinates[0], (int, float))
                else args.coordinates
            )
        if args.box_center:
            if args.kind != "point" or coords:
                raise DomainError("Box center creates a point without explicit coordinates.")
            box = choose(
                [
                    i
                    for i in items
                    if i.kind == "box"
                    and on_slice(i, scope)
                    and (args.target is None or i.target.casefold() == args.target.casefold())
                ]
            )
            coords = [[(a + b) / 2 for a, b in zip(*box.coordinates, strict=True)]]
        if not coords:
            raise DomainError(
                "Provide source voxel coordinates or select a box and explicitly request its "
                "center as an editable starting point."
            )
        if not args.box_center:
            # Validate coordinate provenance, not language intent. The LLM cannot fabricate
            # a position or compute a purported anatomical center from scene metadata.
            supplied = {float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", ctx.message)}
            in_plane = [
                v
                for point in coords
                for axis, v in enumerate(point)
                if len(point) == 2 or scope is None or axis != scope.axis
            ]
            if any(value not in supplied for value in in_plane):
                raise DomainError(
                    "Coordinates must be copied from the user's current request. "
                    "For a point in an existing box, omit coordinates and use box_center=true. "
                    "Otherwise ask for explicit voxel coordinates.",
                    code="invalid_tool_arguments",
                )
        if len(coords) != (2 if args.kind == "box" else 1):
            raise DomainError("Provide one point or two box corners.")
        axes = [
            axis for axis in range(len(asset.spatial_shape)) if scope is None or axis != scope.axis
        ]
        expanded = []
        for point in coords:
            if scope is None and len(point) == 2:
                p = list(point)
            elif scope is not None and len(point) == 2:
                p = [float(scope.index)] * 3
                for axis, value in zip(axes, point, strict=True):
                    p[axis] = value
            elif scope is not None and len(point) == 3:
                p = list(point)
            else:
                raise DomainError(
                    "Provide two in-plane source coordinates or three IJK coordinates."
                )
            if (scope is not None and abs(p[scope.axis] - scope.index) > 0.5) or any(
                v < 0 or v >= size for v, size in zip(p, asset.spatial_shape, strict=True)
            ):
                raise DomainError(
                    "Coordinates must be inside the source image on the current slice."
                )
            if scope is not None:
                p[scope.axis] = float(scope.index)
            expanded.append(p)
        upsert = [
            SpatialObject(
                id=old.id if old else new_id(),
                target=args.target or (old.target if old else ""),
                kind=args.kind,
                coordinates=expanded,
                positive=args.polarity == "positive"
                if args.polarity != "all"
                else old.positive
                if old
                else True,
                selected=old.selected if old else False,
            )
        ]
    action = SpatialEditAction(
        project_id=asset.project_id,
        asset_id=asset.id,
        base_revision=asset.revision,
        slice=scope,
        expected=items,
        upsert=upsert,
        remove=remove,
    )
    return AssistantReply(
        assistant="viewer",
        message=(
            "Updating local spatial prompts. "
            + (
                "The box center is an editable starting point, not a localized anatomical point. "
                if args.box_center
                else ""
            )
            + "Drag hints to refine them before running the spatial model."
        ),
        data=action.model_dump(mode="json"),
    )
