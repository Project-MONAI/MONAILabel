"""Workspace navigation and configuration tools; secrets are entered in protected forms."""

from typing import Literal

from pydantic import Field, JsonValue

from monailabel.core.errors import DomainError
from monailabel.core.evaluation import EvaluationSet, EvaluationSetVersion, ModelSplit
from monailabel.core.models import (
    Asset,
    AssistantReply,
    Contract,
    Evaluation,
    Job,
    Label,
    Learner,
    ModelRecord,
    Project,
    ProjectCreate,
    ReviewDecision,
    Role,
)
from monailabel.server.training_samples import training_sources

from .base import ToolContext, ToolRegistry


class ProjectArgs(Contract):
    name: str = Field(min_length=1, max_length=120)
    labels: list[Label] = Field(
        default_factory=list,
        max_length=32,
        description="Optional structures; Background is included automatically. "
        "Omit or use [] when unspecified.",
    )
    instructions: str = Field(default="", max_length=8000)


class FormArgs(Contract):
    form: Literal["project", "dataset", "dataset-template", "dicom", "model", "credential", "team"]


class ClarifyArgs(Contract):
    question: str = Field(min_length=1, max_length=1000)


class InspectArgs(Contract):
    collection: Literal[
        "projects",
        "models",
        "assets",
        "learners",
        "jobs",
        "review_queue",
        "evaluations",
        "evaluation_sets",
        "evaluation_set_versions",
        "dataset_templates",
        "model_splits",
        "training_sources",
    ]
    offset: int = Field(default=0, ge=0)
    template_id: str | None = Field(
        default=None,
        description="For dataset_templates, inspect one template and all its structures.",
    )


class ViewerArgs(Contract):
    viewer: Literal["slicer", "qupath", "ohif"]


class SelectModelArgs(Contract):
    model_id: str


class ViewerEditArgs(Contract):
    operation: Literal["undo", "redo", "save_draft", "submit"]


class ReviewArgs(Contract):
    verdict: Literal["accepted", "changes_requested"]
    comment: str = Field(default="", max_length=4000)


class CancelArgs(Contract):
    job_id: str


def model_summary(model: ModelRecord) -> dict[str, JsonValue]:
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "label_ids": list[JsonValue](model.label_ids),
        "read_only": model.read_only,
        "spatial_prompts_required": model.provider in {"sam2", "medsam2"},
        "annotation_scope": "medical volume or slice"
        if model.provider == "medsam2"
        else "2D image or selected slice"
        if model.provider == "sam2"
        else None,
        "inherit_targets": model.inherit_targets,
        "preset": model.preset,
        "trained": bool(model.state_key),
        "parent_id": model.parent_id,
        "learner_id": model.learner_id,
        "snapshot_id": model.snapshot_id,
    }


def register(registry: ToolRegistry) -> None:
    ctx = registry.context
    registry.add(
        "clarify_request",
        "Ask for essential missing information before an action. No operation is performed. "
        "Ask only for information the user must supply. Discover internal IDs and catalog "
        "entries through inspect_workspace; never ask the user to supply a template or model ID.",
        ClarifyArgs,
        lambda a: AssistantReply(
            assistant="coordinator",
            message="Before I can proceed: " + a.question,
        ),
        workspace=True,
    )
    registry.add(
        "open_form",
        (
            "Open an editable setup form: project, local file upload (dataset), "
            "sample/template chooser (dataset-template), annotation "
            "model connection, DICOMweb connection/search (dicom), API credential or team. "
            "Never ask for API "
            "key values in chat. Use when configuration details are missing. For a named "
            "sample dataset import, use inspect_workspace(dataset_templates) and "
            "import_dataset_template instead."
        ),
        FormArgs,
        lambda a: form(ctx, a),
        workspace=True,
    )
    registry.add(
        "create_project",
        (
            "Create a named annotation project. Foreground labels are optional "
            "and can be added while annotating. Requires the user's chosen "
            "project name."
        ),
        ProjectArgs,
        lambda a: create_project(ctx, a),
        workspace=True,
    )
    registry.add(
        "inspect_workspace",
        (
            "List accessible projects or current-project models, assets, "
            "trainable learners, jobs, held-out evaluations, pending review queue, or "
            "dataset_templates (available sample/public datasets, IDs, masks, source sections, "
            "modalities and download sizes). Inspect dataset_templates before importing one. "
            "IDs from results "
            "can be used in later tools. Pagination is 50 items."
        ),
        InspectArgs,
        lambda a: inspect(ctx, a),
        workspace=True,
    )
    registry.add(
        "select_annotation_model",
        (
            "Change this conversation/viewer's annotation model selection "
            "using an existing model ID. Does not change project defaults "
            "or the coordinator."
        ),
        SelectModelArgs,
        lambda a: select_model(ctx, a),
    )
    registry.add(
        "open_viewer",
        (
            "Ask the client to prepare/install/reuse and open Slicer, QuPath "
            "or OHIF for the current sample. Actual launch uses the authenticated "
            "viewer API."
        ),
        ViewerArgs,
        lambda a: open_viewer(ctx, a),
    )
    registry.add(
        "viewer_edit",
        (
            "Undo, redo, save a local draft, or submit the current annotation, "
            "only when explicitly requested and supported by the current "
            "viewer."
        ),
        ViewerEditArgs,
        lambda a: viewer_edit(ctx, a),
        requires_asset=True,
        action="edit",
    )
    registry.add(
        "review_annotation",
        (
            "Record Good/accepted (including any corrections) or Bad/needs "
            "changes in the current viewer. Reviewer permission is required. "
            "No decision leaves the sample pending."
        ),
        ReviewArgs,
        lambda a: review(ctx, a),
        requires_asset=True,
        action="review",
    )
    registry.add(
        "cancel_job",
        "Cancel a named running or queued job in the current project.",
        CancelArgs,
        lambda a: cancel(ctx, a),
        action="annotate",
    )


def form(ctx: ToolContext, args: FormArgs) -> AssistantReply:
    if args.form != "project":
        ctx.service.auth.require(ctx.user, ctx.project.id, "manage")
    return AssistantReply(
        assistant="coordinator",
        message={
            "project": "Let's define your project.",
            "dataset": "Choose images and optional labels for shared data, or evaluation only.",
            "dataset-template": "Choose a sample dataset and what to import.",
            "dicom": "Connect to a DICOM server, then choose the series to import.",
            "model": "Connect an annotation model or create a trainable project model.",
            "credential": (
                "Enter your API key in the credential form; keys do not belong in chat history."
            ),
            "team": "Manage users and project roles in Team.",
        }[args.form],
        data={"form": args.form},
    )


def create_project(ctx: ToolContext, args: ProjectArgs) -> AssistantReply:
    labels = args.labels
    if not any(label.id == 0 for label in labels):
        labels = [Label(id=0, name="Background", color="#000000"), *labels]
    project = ctx.service.datasets.create(
        ProjectCreate(name=args.name, labels=labels, instructions=args.instructions)
    )
    ctx.service.auth.add_member(project.id, ctx.user.id, [Role.MANAGER])
    ctx.service.presets.ensure(project.id)
    return AssistantReply(
        assistant="dataset",
        message=(
            f"Created project {project.name}. You can import data and choose annotation models."
        ),
        data={
            "project": ctx.service.store.get(Project, project.id).model_dump(mode="json"),
            "select_project": project.id,
        },
    )


def inspect(ctx: ToolContext, args: InspectArgs) -> AssistantReply:
    store = ctx.service.store
    items: list[JsonValue]
    if args.collection == "projects":
        items = [
            {"id": p.id, "name": p.name}
            for p in store.list(Project)
            if ctx.service.auth.roles(ctx.user, p.id)
        ]
    else:
        project_id = ctx.project.id
        if args.collection == "dataset_templates":
            items = []
            for template in ctx.service.dataset_templates.catalog():
                if args.template_id and template.id != args.template_id:
                    continue
                summary = template.model_dump(
                    mode="json", exclude={"description", "source_url", "license", "cached"}
                )
                if not args.template_id and len(template.targets) > 31:
                    summary.pop("targets")
                    summary["target_count"] = len(template.targets)
                    summary["structure_details"] = (
                        "Inspect this template_id for its full structure list."
                    )
                items.append(summary)
        elif args.collection == "models":
            items = [
                model_summary(m) for m in store.list(ModelRecord, project_id) if not m.archived
            ]
        elif args.collection == "assets":
            items = [
                {
                    "id": a.id,
                    "name": a.name,
                    "kind": a.kind,
                    "shape": list[JsonValue](a.spatial_shape),
                    "revision": a.revision,
                    "split": a.split,
                }
                for a in store.list(Asset, project_id)
            ]
        elif args.collection == "training_sources":
            with store.transaction() as session:
                items = [s.model_dump(mode="json") for s in training_sources(session, project_id)]
        elif args.collection == "model_splits":
            items = [s.model_dump(mode="json") for s in store.list(ModelSplit, project_id)]
        elif args.collection == "learners":
            items = [
                {
                    "id": x.id,
                    "name": x.name,
                    "recipe": x.recipe,
                    "label_ids": list[JsonValue](x.label_ids),
                    "inherit_targets": x.inherit_targets,
                }
                for x in store.list(Learner, project_id)
                if not x.archived
            ]
        elif args.collection == "jobs":
            items = [
                {"id": j.id, "kind": j.kind, "status": j.status, "result": j.result}
                for j in store.list(Job, project_id)
            ]
        elif args.collection in {"evaluation_sets", "evaluation_set_versions"}:
            records = (
                store.list(EvaluationSet, project_id)
                if args.collection == "evaluation_sets"
                else store.list(EvaluationSetVersion, project_id)
            )
            items = [record.model_dump(mode="json") for record in records]
        elif args.collection == "evaluations":
            items = [e.model_dump(mode="json") for e in store.list(Evaluation, project_id)]
        else:
            decisions = {d.annotation_id: d for d in store.list(ReviewDecision, project_id)}
            items = [
                {"asset_id": a.id, "annotation_id": a.annotation_id, "name": a.name}
                for a in store.list(Asset, project_id)
                if a.annotation_id
                and (
                    a.annotation_id not in decisions
                    or decisions[a.annotation_id].verdict == "pending"
                )
            ]
    return AssistantReply(
        assistant="coordinator",
        message=f"{len(items)} {args.collection.replace('_', ' ')} available.",
        data={args.collection: items[args.offset : args.offset + 50], "total": len(items)},
    )


def select_model(ctx: ToolContext, args: SelectModelArgs) -> AssistantReply:
    model = ctx.service.models.get(ctx.project.id, args.model_id)
    return AssistantReply(
        assistant="model",
        message=f"Selected {model.name} for annotation in this conversation.",
        data={"model_id": model.id},
    )


def open_viewer(ctx: ToolContext, args: ViewerArgs) -> AssistantReply:
    if ctx.context.asset_id:
        asset = ctx.asset
        if (args.viewer == "slicer") != (asset.kind == "volume3d") and args.viewer != "ohif":
            raise DomainError("Use Slicer for volumes or QuPath for 2D pathology images.")
    return AssistantReply(
        assistant="viewer",
        message="The client will prepare the viewer and connect it to this project.",
        data={
            "client_action": "use_viewer",
            "viewer": args.viewer,
            "project_id": ctx.project_id,
            "asset_id": ctx.context.asset_id,
        },
    )


def viewer_edit(ctx: ToolContext, args: ViewerEditArgs) -> AssistantReply:
    if args.operation == "submit":
        ctx.service.auth.require(ctx.user, ctx.project.id, "annotate")
    if args.operation not in ctx.context.viewer_actions:
        raise DomainError(
            "This operation is not available through the current viewer's "
            "chat. Use its annotation controls."
        )
    asset = ctx.asset
    return AssistantReply(
        assistant="viewer",
        message=f"Requested {args.operation.replace('_', ' ')} in the viewer.",
        data={
            "client_action": "viewer_edit",
            "operation": args.operation,
            "project_id": ctx.project_id,
            "asset_id": asset.id,
            "base_revision": asset.revision,
        },
    )


def cancel(ctx: ToolContext, args: CancelArgs) -> AssistantReply:
    job = ctx.service.store.get(Job, args.job_id)
    if job.project_id != ctx.project_id:
        raise DomainError("Job belongs to another project.")
    ctx.service.jobs.cancel(job.id)
    return AssistantReply(
        assistant="coordinator",
        message="Requested cancellation of this job.",
        data={"job_id": job.id},
    )


def review(ctx: ToolContext, args: ReviewArgs) -> AssistantReply:
    asset = ctx.asset
    if "review_annotation" not in ctx.context.viewer_actions:
        raise DomainError(
            "Open the sample in an updated viewer to inspect and review its segmentation."
        )
    if not asset.annotation_id:
        raise DomainError("Submit the complete annotation before reviewing it.")
    return AssistantReply(
        assistant="review",
        message="Recording the review decision in your viewer.",
        data={
            "client_action": "review_annotation",
            "project_id": ctx.project_id,
            "asset_id": asset.id,
            "base_revision": asset.revision,
            "annotation_id": asset.annotation_id,
            "verdict": args.verdict,
            "comment": args.comment,
        },
    )
