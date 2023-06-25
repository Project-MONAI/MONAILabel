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
from typing import Dict

from monai.transforms import (
    Transform,
    MapTransform
)

class SAMTransform(Transform):
    """
    Resize longestside (from segment_anything.utils.transforms import ResizeLongestSide
    ??? Looks like the transform is in preprocessing. Not needed in training and inferring
    """
    def __init__(self):
        return

    def __call__(self, filename):
        """
        filename: data path, containing npy_gts and npy_embs, following MedSAM convention
        """
        return

class ToCheck(MapTransform):
    """
    Check dictionary

    """

    def __call__(self, data):
        d: Dict = dict(data)
        print('this is d')
        return d