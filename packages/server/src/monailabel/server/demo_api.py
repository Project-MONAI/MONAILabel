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

"""Small opt-in research telemetry surface; never a public admission service."""

from typing import Literal

from fastapi import APIRouter
from pydantic import Field, JsonValue

from monailabel.core.models import Contract, Job
from monailabel.server.access import Administrator, Principal, Service
from monailabel.server.demo_onboarding import Example
from monailabel.server.demo_training import (
    NOTICE,
    RECIPES,
    STEPS,
    RecordedTraining,
    preview_enabled,
    recorded_example,
)

router = APIRouter(prefix="/api/demo-access")


class StarterSelection(Contract):
    example: Example = "radiology"


@router.post("/starter")
def starter(service: Service, user: Principal, body: StarterSelection | None = None) -> Job | None:
    # No caller-selected project, URL, import count, or privilege override.
    return service.demo_onboarding.start(user, body.example if body else "radiology")


class DemoReset(Contract):
    confirmation: str = Field(max_length=40)
    confirmation_token: str = Field(pattern=r"^[0-9a-f]{64}$")


@router.get("/reset-preview")
def reset_preview(service: Service, user: Administrator) -> dict[str, JsonValue]:
    return service.demo_lifecycle.preview()


@router.post("/reset")
def reset_demo(body: DemoReset, service: Service, user: Administrator) -> dict[str, JsonValue]:
    return service.demo_lifecycle.reset(user, body.confirmation, body.confirmation_token)


class Interaction(Contract):
    kind: Literal["click", "open", "page"]
    # Semantic static IDs only, never element text, form values or URL queries.
    target: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,79}$")
    surface: Literal["workspace", "desktop"] = "workspace"


@router.post("/events", status_code=204)
def interaction(body: Interaction, service: Service, user: Principal) -> None:
    service.audit.record(
        "ui." + body.kind,
        user.id,
        target=body.target,
        surface=body.surface,
        source="browser-untrusted",
    )


@router.get("/audit-status")
def audit_status(service: Service, user: Administrator) -> dict[str, JsonValue]:
    return service.audit.status()


@router.get("/training")
def training_preview(user: Principal) -> dict[str, object]:
    return {
        "enabled": preview_enabled(),
        "notice": NOTICE,
        "steps": STEPS,
        "recipes": RECIPES,
        "job_started": False,
    }


@router.get("/training/{recipe}")
def training_example(recipe: str, user: Principal) -> RecordedTraining:
    return recorded_example(recipe)
