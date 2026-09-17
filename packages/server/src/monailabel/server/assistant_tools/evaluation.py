"""Workspace evaluation-set and pending-review operations."""

from typing import Literal

from pydantic import Field

from monailabel.core.errors import DomainError
from monailabel.core.evaluation import (
    EvaluationSetCreate,
    EvaluationSetDelete,
    EvaluationSetExtend,
    EvaluationSetPublish,
    EvaluationSetUpdate,
)
from monailabel.core.models import (
    Asset,
    AssistantReply,
    BatchReviewRequest,
    Contract,
    ReviewDecision,
    ReviewTarget,
)

from .base import ToolContext, ToolRegistry


class UpdateArgs(EvaluationSetUpdate):
    set_id: str


class ExtendArgs(EvaluationSetExtend):
    set_id: str


class PublishArgs(EvaluationSetPublish):
    set_id: str


class DeleteArgs(EvaluationSetDelete):
    set_id: str


class ReviewArgs(Contract):
    verdict: Literal["accepted", "changes_requested", "pending"]
    scope: Literal["selected", "pending", "accepted", "changes_requested", "all"] = Field(
        description="Required review status filter. Use pending for pending reviews, all for all "
        "saved reviews, or selected only with explicit annotation_ids."
    )
    annotation_ids: list[str] = Field(default_factory=list, max_length=10000)
    dataset_use: Literal["all", "annotation", "evaluation"] = Field(
        default="all",
        description=(
            "Use all unless the CURRENT request explicitly restricts reviews to annotation "
            "or evaluation images. Do not carry an earlier request's dataset filter forward."
        ),
    )
    comment: str = Field(default="", max_length=4000)


def register(registry: ToolRegistry) -> None:
    ctx = registry.context

    def reply(message: str) -> AssistantReply:
        return AssistantReply(assistant="dataset", message=message, data={"page": "datasets"})

    def create(args: EvaluationSetCreate) -> AssistantReply:
        value = ctx.service.evaluation_sets.create(ctx.project.id, args)
        return reply(f"Created {value.name}. Its evaluation cases are kept out of training.")

    def update(args: UpdateArgs) -> AssistantReply:
        value = ctx.service.evaluation_sets.update(
            ctx.project.id,
            args.set_id,
            EvaluationSetUpdate.model_validate(args.model_dump(exclude={"set_id"})),
        )
        return reply(
            f"Updated evaluation set {value.name}. Existing evaluation reservations are preserved."
        )

    def extend(args: ExtendArgs) -> AssistantReply:
        value = ctx.service.evaluation_sets.extend(
            ctx.project.id,
            args.set_id,
            EvaluationSetExtend.model_validate(args.model_dump(exclude={"set_id"})),
        )
        return reply(
            f"Updated the saved case list for {value.name} without reshuffling existing members."
        )

    def publish(args: PublishArgs) -> AssistantReply:
        value = ctx.service.evaluation_sets.publish(
            ctx.project.id,
            args.set_id,
            EvaluationSetPublish.model_validate(args.model_dump(exclude={"set_id"})),
            ctx.user.id,
        )
        return reply(
            f"Evaluation version {value.number} is ready with {len(value.samples)} reviewed cases."
        )

    def delete(args: DeleteArgs) -> AssistantReply:
        ctx.service.evaluation_sets.delete(
            ctx.project.id, args.set_id, EvaluationSetDelete(base_version=args.base_version)
        )
        return reply("Deleted the unused evaluation set. Its cases remain reserved for evaluation.")

    registry.add(
        "create_evaluation_set",
        "Create a named reusable evaluation set from a percentage of project patient "
        "groups. Omit asset_ids for all current samples. auto_update splits future imports;"
        " other eligible cases become training cases. Only when requested; never create a "
        "new set per training run.",
        EvaluationSetCreate,
        create,
        action="manage",
    )
    registry.add(
        "update_evaluation_set",
        "Rename, change percentage, freeze growth (auto_update=false), archive "
        "(archived=true) or restore a set. Inspect evaluation_sets for its current "
        "base_version. Reservations persist.",
        UpdateArgs,
        update,
        action="manage",
    )
    registry.add(
        "extend_evaluation_set",
        "Add specified project samples to an existing set's percentage-based split without "
        "reshuffling existing members.",
        ExtendArgs,
        extend,
        action="manage",
    )
    registry.add(
        "publish_evaluation_set",
        "Save/reuse a fixed evaluation version after all member reference labels are "
        "accepted. label_ids includes background and requested structures; unchanged "
        "references reuse their version.",
        PublishArgs,
        publish,
        action="manage",
    )
    registry.add(
        "delete_evaluation_set",
        "Delete an unused evaluation set only when explicitly requested. Sets with saved "
        "versions must be archived instead; evaluation cases are not released for training.",
        DeleteArgs,
        delete,
        action="manage",
    )
    # Native viewer reviews must submit inspected drafts through their own action.
    if ctx.context.viewer_actions:
        return
    registry.add(
        "review_saved_annotations",
        "Change saved annotation review decisions directly in the workspace, without a "
        "viewer. Good=accepted, Needs changes=changes_requested, Reset to pending=pending. "
        "Use scope=pending for explicitly requested all pending reviews; otherwise selected"
        " annotation_ids. scope=all/accepted/changes_requested only when that group was "
        "explicitly requested. For imported evaluation samples, use dataset_use=evaluation. "
        "An unqualified request for all pending reviews uses dataset_use=all, even after "
        "an earlier request was limited to evaluation samples. Never submits drafts or "
        "unsubmitted predictions.",
        ReviewArgs,
        lambda args: review(ctx, args),
        action="review",
    )


def review(ctx: ToolContext, args: ReviewArgs) -> AssistantReply:
    if args.scope != "selected" and args.annotation_ids:
        raise DomainError("Choose a review scope or a specific selection.")
    project_id = ctx.project.id
    with ctx.service.store.transaction() as session:
        decisions = {d.annotation_id: d for d in session.list(ReviewDecision, project_id)}
        annotations = {
            a.annotation_id
            for a in session.list(Asset, project_id)
            if a.annotation_id
            and (
                args.dataset_use == "all"
                or (a.split == "validation") == (args.dataset_use == "evaluation")
            )
        }
        if args.scope == "selected":
            if not args.annotation_ids or not set(args.annotation_ids) <= annotations:
                raise DomainError("Choose current saved annotations from this project.")
            identifiers = args.annotation_ids
        else:
            identifiers = [
                identifier
                for identifier in annotations
                if args.scope == "all"
                or (decisions[identifier].verdict if identifier in decisions else "pending")
                == args.scope
            ]
        items = [
            ReviewTarget(annotation_id=i, decision_id=decisions[i].id if i in decisions else None)
            for i in identifiers
        ]
    result = (
        ctx.service.reviews.decide_batch(
            project_id,
            BatchReviewRequest(items=items, verdict=args.verdict, comment=args.comment),
            ctx.user,
        )
        if items
        else []
    )
    label = {"accepted": "Good", "changes_requested": "Needs changes", "pending": "Pending review"}[
        args.verdict
    ]
    return AssistantReply(
        assistant="review",
        message=f"Marked {len(result)} saved reviews as {label}.",
        data={"page": "review"},
    )
