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

"""Select native viewer input modes without generating coordinates or invoking inference."""

from pydantic import Field

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    AssistantReply,
    Contract,
    InteractionMode,
    SpatialInteractionAction,
)
from monailabel.providers.spatial import capabilities

from .annotation import AnnotationArgs, annotate
from .base import ToolContext, ToolRegistry


class InteractionArgs(Contract):
    mode: InteractionMode | None = Field(
        default=None,
        description="positive/negative points, box, or navigate. Omit for the model default.",
    )
    target: str | None = Field(default=None, min_length=1, max_length=80)
    model_name: str | None = Field(
        default=None,
        description="Explicitly requested model name, such as nnInteractive or MedSAM2.",
    )


def register(registry: ToolRegistry) -> None:
    registry.add(
        "set_interaction_mode",
        "Activate a persistent native viewer toolbar mode for user clicks/drawing. "
        "Start nnInteractive for spleen: mode=positive, target=Spleen, model_name=nnInteractive. "
        "Switch to negative points means mode=negative, retaining the current target and model. "
        "Draw a spleen box means mode=box, target=Spleen; no coordinates needed. "
        "Stop interaction mode means mode=navigate. Does not run inference or clear hints/masks.",
        InteractionArgs,
        lambda args: set_mode(registry.context, args),
        action="edit",
    )


def set_mode(ctx: ToolContext, args: InteractionArgs) -> AssistantReply:
    if ctx.context.video:
        from .videos import interaction_mode

        return interaction_mode(ctx, args.mode, args.target, args.model_name)
    asset, context = ctx.asset, ctx.context
    if "set_interaction_mode" not in context.viewer_actions:
        raise DomainError("Open an updated Slicer, OHIF or QuPath viewer to place spatial prompts.")
    if context.base_revision is None or (asset.kind == "volume3d" and context.slice is None):
        raise DomainError("Open an image or volume slice first.")
    if context.slice and (
        asset.kind != "volume3d" or context.slice.index >= asset.spatial_shape[context.slice.axis]
    ):
        raise DomainError("Select a slice inside the source image.")
    mode = args.mode
    model_id = ctx.model_id(None, args.model_name) if mode != "navigate" else None
    if mode != "navigate":
        if not model_id:
            raise DomainError("Select an interactive model first.")
        model = ctx.service.models.get(ctx.project.id, model_id)
        spec = capabilities(model)
        if mode is None:
            mode = "positive" if spec and "positive_point" in spec.inputs else "box"
        kind = "box" if mode == "box" else mode + "_point"
        if (
            spec is None
            or kind not in spec.inputs
            or not ctx.service.models.compatible(model, asset)
        ):
            raise DomainError(f"{model.name} does not support {mode} input.")
    target = args.target.strip() if args.target else context.interaction_target
    action = SpatialInteractionAction(
        project_id=asset.project_id,
        asset_id=asset.id,
        base_revision=asset.revision,
        slice=context.slice,
        expected=context.spatial_objects,
        mode=mode,
        target=target,
        model_id=model_id,
    )
    return AssistantReply(
        assistant="viewer",
        message="Returning to navigation; hints and masks are kept."
        if mode == "navigate"
        else f"Activating {mode} prompts{(' for ' + target) if target else ''}. "
        "Click or draw in the viewer, then press Update.",
        data=action.model_dump(mode="json"),
    )


def update(ctx: ToolContext, scope: str) -> AssistantReply:
    """The toolbar uses the same validated annotation action without calling a coordinator."""
    ctx.service.auth.require(ctx.user, ctx.project.id, "edit")
    context = ctx.context
    if context.base_revision is None or not context.interaction_target.strip():
        raise DomainError("Choose or type a target label before updating the mask.")
    if not context.model_id:
        raise DomainError("Select an annotation model first.")
    model = ctx.service.models.get(ctx.project.id, context.model_id)
    if ctx.asset.kind == "volume3d":
        if context.slice is None:
            raise DomainError("Select a source slice first.")
        if scope not in model.annotation_scopes:
            raise DomainError(f"{model.name} does not support this annotation scope.")
    elif scope not in {"full", "selected_region"}:
        raise DomainError("Choose the selected region or full image for this model.")
    return annotate(
        ctx,
        AnnotationArgs.model_validate(
            {
                "targets": [context.interaction_target],
                "model_id": model.id,
                "scope": scope,
            }
        ),
    )
