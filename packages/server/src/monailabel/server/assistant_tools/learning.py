"""Learning tools share immutable datasets, trainers and evaluation gates with the API."""

from typing import Literal

from pydantic import Field, JsonValue

from monailabel.core.errors import DomainError
from monailabel.core.evaluation import EvaluationSet
from monailabel.core.models import (
    AssistantReply,
    Contract,
    EvaluateRequest,
    Evaluation,
    Learner,
    LearnerCreate,
    ModelRecord,
    PromoteRequest,
    SelectionRequest,
    StartTraining,
    TrainingMode,
    TrainingSampleFilter,
    TrainRequest,
)
from monailabel.server.labels import resolve_labels

from .base import Empty, ToolContext, ToolRegistry, require


class LearnerArgs(Contract):
    recipe: Literal["monai-unet", "vista3d"]
    targets: list[str] = Field(default_factory=list, max_length=31)
    name: str | None = Field(default=None, max_length=120)
    parent_model_id: str | None = None
    initialization: Literal["scratch", "fine_tune"] | None = Field(
        default=None,
        description="Omit for the recipe default: fine_tune for VISTA3D, scratch for U-Net.",
    )
    start_now: bool = False
    validation_percentage: int | None = Field(
        default=None,
        ge=0,
        le=50,
        description="Only for start_now. Set 20 for an explicit 80:20 train/evaluation request. "
        "Omit to train accepted samples without creating an evaluation split.",
    )


class NextCasesArgs(SelectionRequest):
    limit: int = Field(default=1, ge=1, le=100)


class TrainArgs(Contract):
    learner_id: str | None = None
    learner_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Exact training setup name from learners when the user names a model. "
        "Overrides the current selection. Prefer this over copying an opaque learner ID.",
    )
    mode: Literal["auto", "scratch", "fine_tune", "continue"] = "auto"
    targets: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=31,
        description="Structure names such as Spleen, never numeric label IDs. "
        "Omit to use the named model's configured structures; supply only for an explicit "
        "request to change training structures.",
    )
    parent_model_id: str | None = Field(
        default=None,
        description="The id of the checkpoint to continue/fine-tune, NOT its parent_id field. "
        "Omit to use the selected model from context.",
    )

    evaluation_set_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Exact fixed evaluation set name from workspace data. Prefer this to its ID.",
    )
    sample_filter: TrainingSampleFilter = Field(
        default_factory=TrainingSampleFilter,
        description="Optional training-only source/image selection and sample limit. "
        "Discover source IDs with inspect_workspace(training_sources). Evaluation is unchanged.",
    )
    validation_percentage: int | None = Field(
        default=None,
        ge=0,
        le=50,
        description="This model's validation percentage (80:20 means 20). "
        "Use 0 to train without evaluation. Omit to retain a saved evaluation choice, "
        "or train without evaluation when none is configured. "
        "Percentage sets belong to this model and grow "
        "as annotations are accepted.",
    )
    evaluation_set_id: str | None = Field(
        default=None,
        description="Named evaluation set to use. References are prepared automatically.",
    )
    evaluation_version_id: str | None = None
    config: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Explicit user-requested settings for this run only; "
        "omitted values use recommended defaults.",
    )


class EvaluateArgs(Contract):
    evaluation_set_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Exact fixed evaluation set name from workspace data.",
    )
    evaluation_set_id: str | None = None
    evaluation_version_id: str | None = None
    candidate_id: str | None = Field(
        default=None,
        description="Exact checkpoint ID only when selecting a particular version. "
        "For a named model prefer candidate_name.",
    )
    baseline_id: str | None = Field(
        default=None,
        description="Exact baseline checkpoint ID only for a particular version. "
        "For a named reference model prefer baseline_name.",
    )
    candidate_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Name of the trained model requested by the user. Prefer this to copying "
        "an ID; the service resolves its latest active version.",
    )
    baseline_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Name of the reference model requested by the user. Prefer this to copying "
        "an ID. Do not substitute the current default model for an explicitly named baseline.",
    )
    snapshot_id: str | None = None


class PromoteArgs(Contract):
    evaluation_id: str | None = None


def register(registry: ToolRegistry) -> None:
    ctx = registry.context
    registry.add(
        "create_learner",
        (
            "Create a separate trainable segmentation model. U-Net uses "
            "monai-unet from scratch for 2D RGB pathology/video or 3D radiology. "
            "VISTA3D fine_tune uses the read-only CT base; never overwrite it. "
            "VISTA3D targets are optional: omit them to inherit the base vocabulary "
            "and choose organs at training time. U-Net requires targets. Set start_now "
            "only when training is requested."
        ),
        LearnerArgs,
        lambda a: create_learner(ctx, a),
        action="manage",
    )
    registry.add(
        "start_training",
        (
            "Train the named (learner_name) or selected project learner, "
            "using validation_percentage=20 for an 80:20 train/evaluation request. "
            "A requested percentage creates this model's split; "
            "no existing evaluation set is needed. Freeze a NEW immutable snapshot "
            "of all currently eligible accepted cases, including newly reviewed data. "
            "Infers scratch vs fine-tune from learner "
            "initialization. To continue a trained version, set mode=continue and "
            "parent_model_id to that checkpoint ID (or use the selected model). "
            "For VISTA3D, targets chooses annotated organs for this run. "
            "When a new inherited setup has no chosen organs, ask which to fine-tune. "
            "Never bypass review or fabricate training "
            "labels."
        ),
        TrainArgs,
        lambda a: train(ctx, a),
        action="manage",
    )
    registry.add(
        "create_snapshot",
        "Freeze accepted image, region or video-frame coverage "
        "and separate held-out source groups.",
        Empty,
        lambda a: snapshot(ctx),
        action="manage",
    )
    registry.add(
        "evaluate_candidate",
        (
            "Start a NEW evaluation job only when requested. To show existing results, use "
            "inspect_workspace(collection=evaluations). Evaluate a trained candidate "
            "against a baseline on the held-out "
            "snapshot or fixed evaluation set; does not promote it. candidate_name and "
            "baseline_name resolve exact model names; a project model name selects its "
            "latest active trained version."
        ),
        EvaluateArgs,
        lambda a: evaluate(ctx, a),
        action="manage",
    )
    registry.add(
        "promote_candidate",
        (
            "Promote only labels that passed an existing evaluation. Requires "
            "an explicit request to promote."
        ),
        PromoteArgs,
        lambda a: promote(ctx, a),
        action="manage",
    )
    registry.add(
        "next_cases",
        "Only rank/select cases for active learning; does NOT segment or submit annotations. "
        "For segmentation of multiple images use annotate_batch instead.",
        NextCasesArgs,
        lambda a: next_cases(ctx, a),
    )


def create_learner(ctx: ToolContext, args: LearnerArgs) -> AssistantReply:
    service, project = ctx.service, ctx.project
    if not args.targets and args.recipe != "vista3d":
        foreground = [label for label in project.labels if label.id]
        if len(foreground) != 1:
            raise DomainError(
                "Choose at least one target for a U-Net training setup.",
                code="invalid_tool_arguments",
            )
        args = args.model_copy(update={"targets": [foreground[0].name]})
    parent = args.parent_model_id
    initialization = args.initialization or ("fine_tune" if args.recipe == "vista3d" else "scratch")
    if initialization == "fine_tune" and not parent and args.recipe == "vista3d":
        service.presets.ensure(project.id)
        parent = next(
            (
                m.id
                for m in service.store.list(ModelRecord, project.id)
                if m.preset == "vista3d" and m.read_only
            ),
            None,
        )
        if not parent:
            raise DomainError("Enable the preloaded VISTA3D model before creating a derived model.")
    if initialization == "scratch" and parent:
        raise DomainError("Training from scratch cannot have initial pretrained weights.")
    if initialization == "fine_tune" and not parent:
        raise DomainError("Select initial weights for fine-tuning.")
    project, ids = resolve_labels(
        service.store, project.id, args.targets, parent, set_defaults=False
    )
    names = [label.name for label in project.labels if label.id in ids]
    learner = service.learning.create(
        project.id,
        LearnerCreate(
            name=args.name
            or (
                ("U-Net" if args.recipe == "monai-unet" else "VISTA3D")
                + (" · " + ", ".join(names) if names else " project model")
            ),
            recipe=args.recipe,
            label_ids=[0] + ids,
            initial_model_id=parent,
            inherit_targets=not args.targets,
        ),
    )
    data = {"learner_id": learner.id, "page": "models", "project": project.model_dump(mode="json")}
    if not args.start_now:
        return AssistantReply(
            assistant="learning",
            message=f"Created {learner.name} as a separate project model setup. "
            + (
                "All supported base targets are inherited. Choose annotated organs "
                "when starting fine-tuning. "
                if not args.targets
                else ""
            )
            + (
                "Weights will initialize from scratch. "
                if not parent
                else "Fine-tuning will copy its initial model. "
            )
            + (
                "Accept annotation reviews, then ask to train this model. "
                "You can request a train/evaluation ratio or a fixed evaluation set."
            ),
            data=data,
        )
    try:
        job = service.learning.start(
            project.id,
            learner.id,
            StartTraining(
                mode=TrainingMode.FINE_TUNE if parent else TrainingMode.SCRATCH,
                parent_model_id=parent,
                validation_percentage=args.validation_percentage,
            ),
        )
    except DomainError as error:
        return AssistantReply(
            assistant="learning",
            message=(
                f"Prepared {learner.name}. {error} Only cases reviewed for all "
                f"requested targets are eligible."
            ),
            data=data,
        )
    return AssistantReply(
        assistant="learning",
        message=(
            f"Training {learner.name} on a filtered, reviewed snapshot. "
            f"Held-out validation cases are excluded from optimization."
        ),
        job_id=job.id,
        data=data,
    )


def evaluation_set_id(ctx: ToolContext, name: str | None, identifier: str | None) -> str | None:
    if name is None:
        return identifier
    matches = [
        item
        for item in ctx.service.store.list(EvaluationSet, ctx.project.id)
        if not item.archived
        and item.name.strip().casefold() == name.strip().casefold()
        and (not identifier or item.id == identifier)
    ]
    if len(matches) != 1:
        raise DomainError(
            "Choose one exact evaluation set name from workspace data, or inspect "
            "evaluation_sets for its ID. No operation was started.",
            code="invalid_tool_arguments",
        )
    return matches[0].id


def train(ctx: ToolContext, args: TrainArgs) -> AssistantReply:
    set_id = evaluation_set_id(ctx, args.evaluation_set_name, args.evaluation_set_id)
    learner_id = args.learner_id or ctx.context.learner_id
    if args.learner_name is not None:
        matches = [
            learner
            for learner in ctx.service.store.list(Learner, ctx.project.id)
            if not learner.archived
            and learner.name.strip().casefold() == args.learner_name.strip().casefold()
        ]
        if args.learner_id:
            matches = [learner for learner in matches if learner.id == args.learner_id]
        if len(matches) != 1:
            return AssistantReply(
                assistant="learning",
                message=(
                    f'Several training setups are named "{args.learner_name}". '
                    "Which one do you mean? Open Start training on the intended setup "
                    "to select it, then refer to this model."
                    if matches
                    else f'No training setup named "{args.learner_name}" matches this request. '
                    "Which setup from Models → Training should I use?"
                ),
            )
        learner_id = matches[0].id
    parent_id = args.parent_model_id
    if args.mode == "fine_tune" and not parent_id and learner_id:
        selected_learner = ctx.service.store.get(Learner, learner_id)
        if selected_learner.project_id != ctx.project_id:
            raise DomainError("Learner belongs to another project.")
        selected_model = (
            ctx.service.models.get(ctx.project.id, ctx.context.model_id)
            if ctx.context.model_id
            else None
        )
        if (
            selected_model
            and selected_model.provider == selected_learner.recipe
            and (
                args.learner_name is None
                or selected_model.learner_id == selected_learner.id
                or selected_model.id == selected_learner.initial_model_id
            )
        ):
            parent_id = selected_model.id
        else:
            parent_id = selected_learner.initial_model_id
    if args.mode in {"continue", "fine_tune"}:
        parent = ctx.service.models.get(
            ctx.project.id, require(parent_id or ctx.context.model_id, "the trained parent model")
        )
        if args.learner_name and parent.learner_id != learner_id and args.mode == "continue":
            raise DomainError("Choose a trained version of the named model to continue.")
        parent_id = parent.id
        learner_id = learner_id or parent.learner_id
    if not learner_id:
        learners = [
            item for item in ctx.service.store.list(Learner, ctx.project.id) if not item.archived
        ]
        if len(learners) == 1:
            learner_id = learners[0].id
    if not learner_id and ctx.project.is_demo:
        job = ctx.service.learning.train(
            ctx.project.id,
            TrainRequest(snapshot_id=require(ctx.context.snapshot_id, "a dataset snapshot")),
        )
    else:
        learner = ctx.service.store.get(Learner, require(learner_id, "a trainable project model"))
        if learner.project_id != ctx.project_id:
            raise DomainError("Learner belongs to another project.")
        mode = (
            (TrainingMode.FINE_TUNE if learner.initial_model_id else TrainingMode.SCRATCH)
            if args.mode == "auto"
            else TrainingMode(args.mode)
        )
        label_ids = None
        if args.targets is not None:
            labels = {label.name.casefold(): label.id for label in ctx.project.labels if label.id}
            missing = [name for name in args.targets if name.casefold() not in labels]
            if missing:
                raise DomainError("Annotate these targets before training: " + ", ".join(missing))
            label_ids = [0] + [labels[name.casefold()] for name in args.targets]
        job = ctx.service.learning.start(
            ctx.project.id,
            learner.id,
            StartTraining(
                sample_filter=args.sample_filter,
                mode=mode,
                label_ids=label_ids,
                config=args.config,
                evaluation_set_id=set_id,
                evaluation_version_id=args.evaluation_version_id,
                validation_percentage=args.validation_percentage
                if not (set_id or args.evaluation_version_id)
                else None,
                parent_model_id=(parent_id or learner.initial_model_id)
                if mode != TrainingMode.SCRATCH
                else None,
            ),
            authorized_by=ctx.user.id,
        )
    return AssistantReply(
        assistant="learning",
        message=(f"Training {learner.name}" if learner_id else "Training")
        + " on an immutable snapshot of accepted cases.",
        job_id=job.id,
        data={"learner_id": learner_id} if learner_id else {},
    )


def snapshot(ctx: ToolContext) -> AssistantReply:
    value = ctx.service.datasets.snapshot(ctx.project.id)
    return AssistantReply(
        assistant="dataset",
        message="Created an immutable snapshot of fully reviewed cases.",
        data={"snapshot_id": value.id},
    )


def evaluate(ctx: ToolContext, args: EvaluateArgs) -> AssistantReply:
    set_id = evaluation_set_id(ctx, args.evaluation_set_name, args.evaluation_set_id)
    set_id = set_id or ctx.context.evaluation_set_id
    version_id = args.evaluation_version_id or (
        ctx.context.evaluation_version_id if not args.snapshot_id and not set_id else None
    )
    job = ctx.service.learning.evaluate(
        ctx.project.id,
        EvaluateRequest(
            evaluation_set_id=set_id if not version_id else None,
            snapshot_id=(args.snapshot_id or ctx.context.snapshot_id)
            if not version_id and not set_id
            else None,
            evaluation_version_id=version_id,
            candidate_id=require(
                ctx.model_id(args.candidate_id, args.candidate_name)
                if args.candidate_name
                else args.candidate_id or ctx.context.model_id,
                "the candidate model",
            ),
            baseline_id=require(
                ctx.model_id(args.baseline_id, args.baseline_name)
                if args.baseline_name
                else args.baseline_id or ctx.context.baseline_id,
                "the baseline model",
            ),
        ),
        authorized_by=ctx.user.id,
    )
    return AssistantReply(
        assistant="learning", message="Started held-out evaluation.", job_id=job.id
    )


def promote(ctx: ToolContext, args: PromoteArgs) -> AssistantReply:
    evaluation = ctx.service.store.get(
        Evaluation, require(args.evaluation_id or ctx.context.evaluation_id, "an evaluation")
    )
    if evaluation.project_id != ctx.project_id:
        raise DomainError("Evaluation belongs to another project.")
    if not evaluation.eligible_labels:
        raise DomainError("No classes passed promotion criteria. Current defaults are retained.")
    value = ctx.service.learning.promote(
        ctx.project.id,
        PromoteRequest(
            evaluation_id=evaluation.id,
            label_ids=evaluation.eligible_labels,
            base_version=ctx.project.version,
        ),
    )
    return AssistantReply(
        assistant="learning",
        message="Promoted the candidate for the classes that passed evaluation.",
        data={"promotion_id": value.id},
    )


def next_cases(ctx: ToolContext, args: SelectionRequest) -> AssistantReply:
    job = ctx.service.selection.select(ctx.project.id, args)
    return AssistantReply(
        assistant="active_learning", message="Selecting the next annotation cases.", job_id=job.id
    )
