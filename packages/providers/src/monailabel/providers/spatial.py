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

"""Capabilities for models requiring viewer spatial prompts."""

from dataclasses import dataclass

from monailabel.core.models import InteractionCapabilities, ModelRecord
from monailabel.providers.sam import MODELS as SAM_MODELS


@dataclass(frozen=True)
class SpatialModel:
    name: str
    interaction: InteractionCapabilities


MODELS = {
    **{
        key: SpatialModel(
            spec.name,
            InteractionCapabilities(
                inputs={"positive_point": "slice", "negative_point": "slice", "box": "slice"},
                output_scopes=["current_slice", "full"] if spec.volume else ["current_slice"],
                volume_only=spec.volume,
                video_scopes=["frame", "range"] if key == "sam2" else [],
            ),
        )
        for key, spec in SAM_MODELS.items()
    },
    "nninteractive": SpatialModel(
        "nnInteractive",
        InteractionCapabilities(
            inputs={"positive_point": "volume", "negative_point": "volume", "box": "slice"},
            output_scopes=["current_slice", "full"],
            prompt_scope="volume",
            volume_only=True,
            intensity_window=False,
        ),
    ),
}


def capabilities(model: ModelRecord) -> InteractionCapabilities | None:
    spec = MODELS.get(model.provider)
    return spec.interaction if spec else None
