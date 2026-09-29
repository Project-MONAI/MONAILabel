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

"""Pinned VISTA3D provenance and automatic CT target vocabulary; no framework imports.

Source: MONAI/vista3d bundle 0.5.11, docs/labels.json and configs/inference.json.
Uses the bundle's 117-class default automatic set plus lung tumor (class 23),
which upstream supports through an explicit class prompt (docs/inference.md).
Other separately requested classes, combined subclasses and legacy IDs are excluded.
"""

import json
from functools import cache
from importlib.resources import files

from monailabel.core.errors import DomainError
from monailabel.core.models import Label

REVISION = "c6dbe159632a4767696e09f91d74d729b82e73e6"
CHECKSUM = "c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b"
WEIGHT_BYTES = 871_970_895
WEIGHT_URL = f"https://huggingface.co/MONAI/vista3d/resolve/{REVISION}/models/model.pt"


@cache
def targets() -> dict[str, int]:
    data = json.loads(files(__package__).joinpath("resources/vista3d_labels.json").read_text())
    return {name.casefold(): index for name, index in data.items()}


def mapping(labels: list[Label]) -> dict[int, int]:
    names = targets()
    unknown = [label.name for label in labels if label.id and label.name.casefold() not in names]
    if unknown:
        raise DomainError(
            "VISTA3D automatic CT segmentation does not support these targets: "
            + ", ".join(unknown)
            + ". Select another model or use a supported anatomical name."
        )
    return {label.id: names[label.name.casefold()] for label in labels if label.id}
