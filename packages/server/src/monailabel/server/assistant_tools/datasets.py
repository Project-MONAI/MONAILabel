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

"""Chat access to the same curated dataset imports as the workspace form."""

from pydantic import Field

from monailabel.core.dataset_templates import DatasetTemplateImport
from monailabel.core.models import AssistantReply

from .base import ToolContext, ToolRegistry


class TemplateArgs(DatasetTemplateImport):
    all_samples: bool = Field(
        default=False, description="True only when all images were requested; ignores limit."
    )


class TemplateImportArgs(TemplateArgs):
    limit: int = Field(default=5, ge=1, le=2000, description="Maximum samples; default 5.")


class TemplateSplitArgs(TemplateArgs):
    include_masks: bool = Field(
        default=False,
        description="Import reference masks for the ANNOTATION/TRAINING portion only. "
        "True for 'import with labels' or 'labeled cases'; false for 'images only' "
        "or labels requested only for evaluation. "
        "The evaluation portion ALWAYS receives its reference masks independently.",
    )
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
        "Video templates import one clip for CVAT annotation: split=pool, no masks or "
        "evaluation split. They do not enter image training. "
        "Honor the requested image count (limit); all_samples=true means all, "
        "omitted count means 5. "
        "For a percentage split between annotation and independent evaluation, "
        "use import_dataset_split instead. "
        "Default: images only, source training section, annotation/training "
        "(split=pool, including imports intended for later training). "
        "For evaluation requests, set split=validation: images AND supplied labels are "
        "automatically imported from the labeled training section and reserved for evaluation. "
        "Published evaluation references are accepted on import. "
        "For other uses, include labels only when requested; those labels need review. "
        "Only start when the user requests an import, not when merely browsing options.",
        TemplateImportArgs,
        lambda args: import_template(registry.context, args),
        action="manage",
    )
    registry.add(
        "import_dataset_split",
        "Import BOTH portions of a sample dataset in one job: a percentage of images WITH "
        "labels reserved for independent evaluation, and the remainder for annotation/training. "
        "'Import with labels; reserve 20% for evaluation' uses evaluation_percentage=20, "
        "include_masks=true. '80% images only, 20% with evaluation labels' uses "
        "evaluation_percentage=20, include_masks=false. Both use split=pool. "
        "Omit limit (all cases) unless the user requests a total sample count. "
        "Do not import just the evaluation portion or use the five-image quick-start default. "
        "Use an exact template_id from inspect_workspace(dataset_templates). "
        "Uses the labeled source training section; evaluations always include provided labels. "
        "include_masks controls labels for the annotation/training remainder only. "
        "Evaluation count rounds up to whole cases, leaving at least one annotation case. "
        "Published evaluation references are accepted on import; training labels need review. "
        "Existing annotations, review decisions and reservations are protected.",
        TemplateSplitArgs,
        lambda args: import_template(registry.context, args),
        action="manage",
    )


def import_template(ctx: ToolContext, args: TemplateArgs) -> AssistantReply:
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
    if template.kind == "video":
        message = (
            f"Started importing one video from {template.name} into {project.name}. "
            "Progress appears in Activity and the clip appears in Datasets. "
            "Open CVAT to annotate it, save there, then submit the saved tracks for review. "
            "Existing annotations and drafts are preserved."
        )
    elif request.evaluation_percentage is not None:
        percentage = request.evaluation_percentage
        message = (
            f"Started importing {count} cases from {template.name} into {project.name}: "
            f"{100 - percentage:g}% {content} for annotation/training and "
            f"{percentage:g}% images with labels for evaluation. "
            "Evaluation rounds up to whole cases and stays excluded from training. "
            "Published evaluation references are ready to use; training labels need review. "
            "Progress appears in Activity and imported cases "
            "appear in Datasets. Existing annotations are preserved."
        )
    else:
        message = (
            f"Started importing {count} {content} from {template.name} into {project.name}. "
            "Progress and imported images will appear in Datasets."
            + (
                " Published references are accepted on import, reserved for evaluation "
                "and excluded from training."
                if request.split == "validation"
                else " Reference masks will be pending review."
                if request.include_masks
                else ""
            )
        )
    return AssistantReply(
        assistant="dataset",
        message=message,
        job_id=job.id,
        data={"page": "datasets", "template_id": template.id},
    )
