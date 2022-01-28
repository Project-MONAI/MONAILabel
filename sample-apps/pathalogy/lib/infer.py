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

import numpy as np
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CastToTypeD,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import BoundingBoxd, Restored


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="generic",
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation over 3D Images",
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
        ]

    def inferer(self):
        return SimpleInferer()  # SlidingWindowInferer(roi_size=(512, 512), sw_batch_size=4, overlap=0.25)

    def post_transforms(self):
        return [
            # Activationsd(keys="pred", sigmoid=True),
            # AsDiscreted(keys="pred", argmax=True),
            # SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
        ]
