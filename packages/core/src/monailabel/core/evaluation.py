"""Reusable evaluation policies, immutable reference versions, and reservations."""

from typing import Literal

from pydantic import Field

from monailabel.core.models import Contract, Label, Record, Sample, new_id


class ModelSplit(Record):
    project_id: str
    learner_id: str
    validation_percentage: int = 20
    validation_mode: Literal["percentage", "fixed", "none"] = "percentage"
    label_ids: list[int] = Field(default_factory=list)
    version: int = 0
    training_groups: list[str] = Field(default_factory=list)
    validation_groups: list[str] = Field(default_factory=list)
    training_image_keys: list[str] = Field(default_factory=list)
    validation_image_keys: list[str] = Field(default_factory=list)


class EvaluationSetCreate(Contract):
    name: str = Field(min_length=1, max_length=120)
    percentage: int = Field(default=20, ge=1, le=100)
    auto_update: bool = True
    asset_ids: list[str] | None = Field(default=None, max_length=10000)


class EvaluationSet(Record):
    project_id: str
    name: str
    percentage: int = 20
    auto_update: bool = True
    archived: bool = False
    version: int = 0
    seed: str = Field(default_factory=new_id)
    cohort_groups: list[str] = Field(default_factory=list)
    member_groups: list[str] = Field(default_factory=list)
    latest_version_id: str | None = None


class EvaluationSetUpdate(Contract):
    base_version: int = Field(ge=0)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    percentage: int | None = Field(default=None, ge=1, le=100)
    auto_update: bool | None = None
    archived: bool | None = None


class EvaluationSetExtend(Contract):
    base_version: int = Field(ge=0)
    asset_ids: list[str] = Field(min_length=1, max_length=10000)
    include_all: bool = False


class EvaluationSetPublish(Contract):
    base_version: int = Field(ge=0)
    label_ids: list[int] = Field(min_length=2, max_length=32)


class EvaluationSetDelete(Contract):
    base_version: int = Field(ge=0)


class EvaluationSetVersion(Record):
    project_id: str
    evaluation_set_id: str
    number: int
    protocol_version: int
    labels: list[Label]
    samples: list[Sample]
    published_by: str


class EvaluationReservation(Record):
    project_id: str
    group_id: str
    image_keys: list[str]
