# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict

import numpy as np
from monai.transforms import RandomizableTransform

logger = logging.getLogger(__name__)


# You can write your transforms here... which can be used in your train/infer tasks
class Random2DSlice(RandomizableTransform):
    def __init__(self, image: str = "image", label: str = "label"):
        super().__init__()

        self.image = image
        self.label = label

    def __call__(self, data):
        d: Dict = dict(data)
        image = d[self.image]
        label = d[self.label]

        if len(image.shape) and len(label.shape) != 3:  # only for 3D
            raise ValueError("Only supports label with shape DHW!")

        sids = []
        for sid in range(label.shape[0]):
            if np.sum(label[sid]) != 0:
                sids.append(sid)

        sid = self.R.choice(sids, replace=False)
        d[self.image] = image[sid]
        d[self.label] = label[sid]
        return d
