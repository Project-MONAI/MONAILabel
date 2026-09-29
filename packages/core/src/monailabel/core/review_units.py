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

"""Independently submitted regions and frame ranges within an immutable source."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from monailabel.core.models import Contract, DecisionRequest, ImageRegion, Record


class RegionScope(Contract):
    kind: Literal["region"] = "region"
    region: ImageRegion


class FrameScope(Contract):
    kind: Literal["frames"] = "frames"
    start: int = Field(strict=True, ge=0)
    stop: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.stop <= self.start:
            raise ValueError("A frame range requires start < stop (stop is exclusive).")
        return self


UnitScope = Annotated[RegionScope | FrameScope, Field(discriminator="kind")]


class ReviewUnit(Record):
    project_id: str
    asset_id: str
    name: str
    scope: UnitScope
    label_ids: list[int]
    revision: int = 0
    annotation_id: str | None = None


class UnitAnnotation(Record):
    project_id: str
    asset_id: str
    unit_id: str
    revision: int
    source_revision: int
    source_annotation_id: str
    scope: UnitScope
    covered_labels: list[int]
    image_key: str
    mask_key: str
    created_by: str


class UnitDecision(DecisionRequest):
    base_revision: int = Field(ge=1)
    decision_id: str | None = None
