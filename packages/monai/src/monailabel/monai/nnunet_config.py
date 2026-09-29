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

"""Settings for one automatically planned, single-channel 3D nnU-Net."""

from typing import Literal

from pydantic import Field

from monailabel.core.models import Contract


class NNUNetConfig(Contract):
    modality: Literal["CT", "MRI"]
    epochs: int = Field(default=20, ge=1, le=1000)
    steps_per_epoch: int = Field(default=32, ge=1, le=1000)
    learning_rate: float = Field(default=0.01, gt=0, le=0.1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
