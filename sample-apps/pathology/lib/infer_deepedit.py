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
from typing import Callable, Sequence

import numpy as np
from monai.apps.deepgrow.transforms import AddGuidanceSignald
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    EnsureTyped,
    ScaleIntensityRangeD,
    SqueezeDimd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferType

from .infer import InferSegmentation
from .transforms import AddClickGuidanced, FilterImaged, FindContoursd, LoadImagePatchd, PostFilterLabeld

logger = logging.getLogger(__name__)


class InferDeepedit(InferSegmentation):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(256, 256),
        type=InferType.DEEPEDIT,
        labels=None,
        dimension=2,
        description="A pre-trained interaction/deepedit model for Nuclei (Pathology)",
    ):
        super().__init__(
            path=path,
            network=network,
            roi_size=roi_size,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self, data=None):
        return [
            LoadImagePatchd(keys="image", conversion="RGB", dtype=np.uint8),
            FilterImaged(keys="image"),
            AsChannelFirstd(keys="image"),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            AddClickGuidanced(image="image", guidance="guidance"),
            AddGuidanceSignald(image="image", guidance="guidance", number_intensity_ch=3),
            EnsureTyped(keys="image"),
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            SqueezeDimd(keys="pred"),
            ToNumpyd(keys=("image", "pred")),
            PostFilterLabeld(keys="pred", image="image"),
            FindContoursd(keys="pred", labels=self.labels),
        ]
