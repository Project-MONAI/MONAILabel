"""Dataset catalog and import choices, independent of download or storage mechanisms."""

from typing import Any, Literal

from pydantic import Field, model_validator

from monailabel.core.models import Contract, Split


class DatasetTemplate(Contract):
    id: str
    name: str
    category: Literal["Radiology", "Pathology", "Video"]
    kind: Literal["image", "video"] = "image"
    description: str
    source_url: str
    license: str
    download_bytes: int = 0
    has_masks: bool = False
    targets: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=lambda: ["training"])
    importable: bool = True
    cached: bool = False


class DatasetTemplateImport(Contract):
    template_id: str = Field(description="Exact ID from the sample dataset catalog.")
    include_masks: bool = Field(
        default=False,
        description="Include labels when requested. Evaluation always includes provided labels.",
    )
    targets: list[str] = Field(default_factory=list, max_length=31)
    split: Split = Field(
        default=Split.POOL,
        description="Project use: pool for annotation (default), train for training, "
        "validation for evaluation. Independent of the source dataset section.",
    )
    section: Literal["training", "test"] = Field(
        default="training",
        description="Source dataset section; use training unless test images were requested.",
    )
    offset: int = Field(default=0, ge=0, le=10000)
    limit: int | None = Field(
        default=5,
        ge=1,
        le=2000,
        description="Maximum samples to import; null imports all samples in the selected section.",
    )
    channel: int = Field(default=0, ge=0, le=3)
    evaluation_set_id: str | None = Field(
        default=None,
        description="Optional existing evaluation set; otherwise reuse a named template set.",
    )
    evaluation_percentage: float | None = Field(
        default=None,
        gt=0,
        lt=100,
        description="For a combined import, reserve this percentage with labels for evaluation; "
        "the rest is for annotation/training. include_masks applies to that remainder.",
    )

    @model_validator(mode="before")
    @classmethod
    def evaluation_includes_labels(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("evaluation_percentage") is not None:
            if value.get("split", Split.POOL) != Split.POOL or value.get("section") == "test":
                raise ValueError(
                    "A combined import uses the labeled source section and annotation/training "
                    "as its default use; evaluation cases are reserved separately."
                )
            return value
        if value.get("split") == Split.VALIDATION:
            if value.get("section") == "test":
                raise ValueError(
                    "Evaluation needs labels. Use the dataset's labeled training section."
                )
            return value | {"include_masks": True}
        if value.get("evaluation_set_id"):
            raise ValueError("Choose Evaluation to import into an evaluation set.")
        return value
