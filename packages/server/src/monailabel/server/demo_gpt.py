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

"""Host-approved GPT annotation choices, never visitor-supplied connections."""

import os

from pydantic import JsonValue

from monailabel.core.models import ModelRecord
from monailabel.server.storage import Session

MODELS = {
    "demo-gpt-sol": ("GPT-6.1 Sol · demo annotation", "gpt-6.1-sol"),
    "demo-gpt-astra": ("GPT-6 Astra · demo comparison", "gpt-6-astra"),
}
NOTICE = (
    "GPT features send conversation context and selected annotation images to OpenAI, "
    "using the host's paid API account. Use public demo data only. GPT annotation is an "
    "experimental draft, not a clinically validated segmentation; inspect and correct it."
)


def enabled() -> bool:
    return os.environ.get("MONAILABEL_DEMO_GPT") == "1" and bool(
        os.environ.get("OPENAI_API_KEY", "").strip()
    )


def configuration(identifier: str) -> dict[str, JsonValue]:
    return {
        "url": "https://api.openai.com/v1/responses",
        "model": identifier,
        "token_env": "OPENAI_API_KEY",
        "max_output_tokens": 8192,
        "reasoning_effort": "low",
        "timeout": 180,
    }


def approved(model: ModelRecord) -> bool:
    spec = MODELS.get(model.preset or "")
    return bool(
        enabled()
        and spec
        and not model.archived
        and model.provider == "openai-polygons"
        and model.read_only
        and model.config == configuration(spec[1])
    )


def ensure(session: Session, project_id: str) -> None:
    """Add metadata only; do not run inference or change project defaults/data."""
    if not enabled():
        return
    existing = session.list(ModelRecord, project_id)
    for preset, (name, identifier) in MODELS.items():
        if any(model.preset == preset for model in existing):
            continue  # Never overwrite an existing connection or its history.
        session.insert(
            ModelRecord(
                project_id=project_id,
                name=name,
                provider="openai-polygons",
                label_ids=[0],
                preset=preset,
                read_only=True,
                config=configuration(identifier),
            )
        )
