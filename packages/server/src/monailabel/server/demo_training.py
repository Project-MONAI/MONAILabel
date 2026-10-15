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

"""Explicit, read-only training demonstrations; never simulated successful jobs."""

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, ValidationError, model_validator

from monailabel.core.errors import DomainError
from monailabel.core.models import Contract

NOTICE = (
    "Training preview — no training will run. Live training is disabled in this shared demo "
    "to keep resources available for everyone. Explore Models → Training → Recorded examples "
    "to view previously completed runs. Your current corrections were not used to produce "
    "those examples. No job has been queued."
)
STEPS = [
    "Select reviewed annotations and keep evaluation patients separate.",
    "Freeze the selected image and annotation revisions in a training snapshot.",
    "Prepare the images and labels; nnU-Net also plans its network and preprocessing.",
    "Train from scratch, fine-tune initial weights, or continue a compatible checkpoint.",
    "Save a separate model version without replacing the shared starting weights.",
    "Evaluate on held-out references when available, then inspect predictions in a viewer.",
]
RECIPES = ("nnunet-v2", "vista3d")
MAX_EXAMPLE_BYTES = 2 * 1024 * 1024


def preview_enabled() -> bool:
    return os.environ.get("MONAILABEL_DEMO_TRAINING_PREVIEW") == "1"


def require_live_training() -> None:
    if preview_enabled():
        raise DomainError(NOTICE, code="demo_training_preview", status=403)


class RecordedTraining(Contract):
    """Operator-exported public-data evidence, not a visitor's job or checkpoint import."""

    schema_version: Literal[1] = 1
    presentation: Literal["recorded_run"] = "recorded_run"
    recipe: Literal["nnunet-v2", "vista3d"]
    name: str = Field(min_length=1, max_length=120)
    dataset: str = Field(min_length=1, max_length=200)
    completed_at: datetime
    duration_seconds: Annotated[FiniteFloat, Field(ge=0)]
    training_cases: int = Field(ge=1, le=10000)
    evaluation_cases: int = Field(ge=0, le=10000)
    epochs: int = Field(ge=1, le=1000)
    steps_per_epoch: int = Field(ge=1, le=1000)
    initial_loss: FiniteFloat | None = None
    final_loss: FiniteFloat | None = None
    mean_dice: Annotated[FiniteFloat, Field(ge=0, le=1)] | None = None
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    caveats: list[Annotated[str, Field(max_length=2000)]] = Field(max_length=20)
    log: str = Field(max_length=500000)

    @model_validator(mode="after")
    def held_out_score(self) -> Self:
        if self.mean_dice is not None and not self.evaluation_cases:
            raise ValueError("A held-out score needs held-out evaluation cases.")
        if self.completed_at.tzinfo is None:
            raise ValueError("The recorded completion time must include a timezone.")
        return self


def recorded_example(recipe: str) -> RecordedTraining:
    location = os.environ.get("MONAILABEL_DEMO_TRAINING_EXAMPLES")
    if recipe not in RECIPES or not location:
        raise DomainError("This recorded training example is not prepared yet.", status=404)
    root = Path(location).resolve()
    path = root / (recipe + ".json")
    try:
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Example must be a regular file in the operator's catalog.")
        if path.exists() and not path.is_file():
            raise ValueError("Example must be a regular file.")
        with path.open("rb") as stream:
            content = stream.read(MAX_EXAMPLE_BYTES + 1)
        if len(content) > MAX_EXAMPLE_BYTES:
            raise ValueError("Example exceeds its display limit.")
        result = RecordedTraining.model_validate_json(content)
        if result.recipe != recipe:
            raise ValueError("Example recipe mismatch.")
        return result
    except FileNotFoundError as exc:
        raise DomainError(
            "This recorded training example is not prepared yet.", status=404
        ) from exc
    except (OSError, ValueError, ValidationError) as exc:
        raise DomainError(
            "The host must verify this recorded training example.", status=503
        ) from exc
