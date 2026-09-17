"""Chat access to the same curated dataset imports as the workspace form."""

from pydantic import Field

from monailabel.core.dataset_templates import DatasetTemplateImport
from monailabel.core.models import AssistantReply

from .base import ToolContext, ToolRegistry


class TemplateImportArgs(DatasetTemplateImport):
    limit: int = Field(default=5, ge=1, le=2000, description="Number of images; default 5.")
    all_samples: bool = Field(
        default=False, description="True only when all images were requested; ignores limit."
    )


class TemplateSplitArgs(DatasetTemplateImport):
    evaluation_percentage: float = Field(
        gt=0, lt=100, description="Percentage reserved for evaluation; 80/20 means 20."
    )
    limit: int | None = Field(
        default=None,
        ge=2,
        le=2000,
        description="Total source cases across BOTH portions. Omit for the whole source section.",
    )


def register(registry: ToolRegistry) -> None:
    registry.add(
        "import_dataset_template",
        "Download/reuse and import a sample/public dataset into the current project. "
        "Use an exact importable template_id from inspect_workspace(dataset_templates). "
        "A named template supplies its source URL; no upload/path is needed. "
        "Honor the requested image count (limit); all_samples=true means all, "
        "omitted count means 5. "
        "For a percentage split between annotation and independent evaluation, "
        "use import_dataset_split instead. "
        "Default: images only, source training section, annotation/training "
        "(split=pool, including imports intended for later training). "
        "For evaluation requests, set split=validation: images AND supplied labels are "
        "automatically imported from the labeled training section and reserved for evaluation. "
        "For other uses, include labels only when requested. Imported labels need review. "
        "Only start when the user requests an import, not when merely browsing options.",
        TemplateImportArgs,
        lambda args: import_template(registry.context, args),
        action="manage",
    )
    registry.add(
        "import_dataset_split",
        "Import BOTH portions of a sample dataset in one job: a percentage of images WITH "
        "labels reserved for independent evaluation, and the remainder for annotation/training. "
        "Use this for '80% images only for annotation and remaining 20% images+labels for "
        "evaluation': evaluation_percentage=20, include_masks=false, split=pool. "
        "Omit limit (all cases) unless the user requests a total sample count. "
        "Do not import just the evaluation portion or use the five-image quick-start default. "
        "Use an exact template_id from inspect_workspace(dataset_templates). "
        "Uses the labeled source training section; evaluations always include provided labels. "
        "include_masks controls labels for the annotation/training remainder only. "
        "Evaluation count rounds up to whole cases, leaving at least one annotation case. "
        "Existing annotations/reservations are protected. Labels remain pending review.",
        TemplateSplitArgs,
        lambda args: start_import(registry.context, args),
        action="manage",
    )


def import_template(ctx: ToolContext, args: TemplateImportArgs) -> AssistantReply:
    request = DatasetTemplateImport.model_validate(
        args.model_dump(exclude={"all_samples"})
        | {"limit": None if args.all_samples else args.limit}
    )
    return start_import(ctx, request)


def start_import(ctx: ToolContext, request: DatasetTemplateImport) -> AssistantReply:
    project = ctx.project
    job = ctx.service.dataset_templates.start(project.id, request, ctx.user.id)
    template = next(
        t for t in ctx.service.dataset_templates.catalog() if t.id == request.template_id
    )
    count = "all" if request.limit is None else f"up to {request.limit}"
    content = "images with reference masks" if request.include_masks else "images only"
    if request.evaluation_percentage is not None:
        percentage = request.evaluation_percentage
        message = (
            f"Started importing {count} cases from {template.name} into {project.name}: "
            f"{100 - percentage:g}% {content} for annotation/training and "
            f"{percentage:g}% images with labels for evaluation. "
            "Evaluation rounds up to whole cases and stays excluded from training. "
            "Labels will be pending review. Progress appears in Activity and imported cases "
            "appear in Datasets. Existing annotations are preserved."
        )
    else:
        message = (
            f"Started importing {count} {content} from {template.name} into {project.name}. "
            "Progress and imported images will appear in Datasets."
            + (" Reference masks will be pending review." if request.include_masks else "")
            + (
                " These samples are reserved for evaluation and excluded from training."
                if request.split == "validation"
                else ""
            )
        )
    return AssistantReply(
        assistant="dataset",
        message=message,
        job_id=job.id,
        data={"page": "datasets", "template_id": template.id},
    )
