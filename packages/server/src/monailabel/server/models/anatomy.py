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

"""Shared target handling for pretrained anatomical models and their derived versions."""

from monailabel.core.models import Label, ModelRecord
from monailabel.providers.vista3d import mapping as vista_mapping
from monailabel.providers.vista3d import targets as vista_targets
from monailabel.totalsegmentator.catalog import MODELS as TOTAL_MODELS
from monailabel.totalsegmentator.catalog import mapping as total_mapping
from monailabel.totalsegmentator.catalog import normalize
from monailabel.totalsegmentator.catalog import targets as total_targets

ANATOMY_MODELS = {"vista3d", *TOTAL_MODELS}


def inherited(model: ModelRecord) -> bool:
    return model.provider in ANATOMY_MODELS and (model.read_only or model.inherit_targets)


def targets(provider: str) -> dict[str, int]:
    return vista_targets() if provider == "vista3d" else total_targets(provider)


def supports(provider: str, name: str) -> bool:
    if provider == "vista3d":
        return name.casefold() in vista_targets()
    return normalize(name) in {normalize(target) for target in targets(provider)}


def mapping(provider: str, labels: list[Label]) -> dict[int, int]:
    return vista_mapping(labels) if provider == "vista3d" else total_mapping(provider, labels)
