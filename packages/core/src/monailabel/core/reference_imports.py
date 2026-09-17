"""Explicitly imported labels and optional evaluation-only membership."""

from typing import Literal

from pydantic import Field, model_validator

from monailabel.core.evaluation import EvaluationSet
from monailabel.core.models import Contract, Record, Split


class LabelImportRequest(Contract):
    split: Split = Split.POOL
    evaluation_set_id: str | None = None
    evaluation_set_name: str | None = Field(default=None, min_length=1, max_length=120)
    base_version: int | None = Field(default=None, ge=0)
    group_id: str | None = Field(default=None, min_length=1, max_length=200)
    labels: dict[int, str] = Field(min_length=1, max_length=31)
    reviewed: bool = False
    source: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def check_set(self) -> "LabelImportRequest":
        if self.split == Split.VALIDATION and bool(self.evaluation_set_id) == bool(
            self.evaluation_set_name
        ):
            raise ValueError("Choose an evaluation set or give a new set a name.")
        if self.split != Split.VALIDATION and (
            self.evaluation_set_id or self.evaluation_set_name or self.base_version is not None
        ):
            raise ValueError("Evaluation sets can only be used for evaluation imports.")
        if self.evaluation_set_id and self.base_version is None:
            raise ValueError("Refresh the evaluation set before importing.")
        if any(k < 1 or k > 65535 or not v.strip() or len(v) > 80 for k, v in self.labels.items()):
            raise ValueError(
                "Use foreground label values 1–65535 and structure names up to 80 characters."
            )
        return self


class EvaluationImportRequest(LabelImportRequest):
    split: Literal[Split.VALIDATION] = Split.VALIDATION


class ReferenceImport(Record):
    project_id: str
    asset_id: str
    annotation_id: str
    evaluation_set_id: str | None = None
    image_name: str
    label_name: str
    source_label_key: str
    label_mapping: dict[int, int]
    reviewed: bool
    imported_by: str
    source: str = ""


class EvaluationImportResult(Contract):
    asset_id: str
    annotation_id: str
    covered_labels: list[int]
    evaluation_set: EvaluationSet | None = None
