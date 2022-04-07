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
from typing import Dict

from monai.transforms import MapTransform


class GetSingleModalityBRATSd(MapTransform):
    """
    Gets one modality

    "0": "FLAIR",
    "1": "T1w",
    "2": "t1gd",
    "3": "T2w"

    """

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                # TRANSFORM IN PROGRESS - SHOULD AFFINE AND ORIGINAL BE CHANGED??

                # Output is only one channel
                # Get T1 Gadolinium. Better to describe brain tumour. FLAIR is better for edema (swelling in the brain)
                d[key] = d[key][..., 2]
                # d[key] = d[key][None] # Add dimension
            else:
                print("Transform 'GetSingleModalityBRATSd' only works for the image key")

        return d
