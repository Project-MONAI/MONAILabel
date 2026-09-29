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

"""Pinned upstream tasks and class IDs; importing the catalog never loads a network."""

import json
from functools import cache
from importlib.resources import files

from monailabel.core.errors import DomainError
from monailabel.core.models import Label

VERSION = "2.18.0"
MODELS = {
    "totalsegmentator-ct": ("TotalSegmentator CT", "total"),
    "totalsegmentator-mr": ("TotalSegmentator MRI", "total_mr"),
}
DOCUMENTATION = f"https://github.com/wasserth/TotalSegmentator/tree/v{VERSION}"


def normalize(name: str) -> str:
    return "_".join(name.casefold().replace("_", " ").split())


@cache
def targets(provider: str) -> dict[str, int]:
    if provider not in MODELS:
        raise DomainError("Choose TotalSegmentator CT or MRI.")
    data = json.loads(files(__package__).joinpath("resources/labels.json").read_text())
    return {str(name): int(index) for name, index in data[MODELS[provider][1]].items()}


def mapping(provider: str, labels: list[Label]) -> dict[int, int]:
    names = {normalize(name): index for name, index in targets(provider).items()}
    missing = [label.name for label in labels if label.id and normalize(label.name) not in names]
    if missing:
        raise DomainError(f"{MODELS[provider][0]} does not support: " + ", ".join(missing))
    result = {label.id: names[normalize(label.name)] for label in labels if label.id}
    if len(set(result.values())) != len(result):
        raise DomainError("Use only one project label for each anatomical structure.")
    return result
