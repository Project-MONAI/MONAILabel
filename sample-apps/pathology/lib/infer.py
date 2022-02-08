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

import numpy as np
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityd,
    SqueezeDimd,
    ToNumpyd,
    Transposed,
)
from monai.utils import BlendMode

from monailabel.interfaces.tasks.infer import InferTask, InferType

logger = logging.getLogger(__name__)


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        patch_size=(256, 256),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        sliding_window=True,
        description="A pre-trained model Pathology",
    ):
        self.patch_size = patch_size
        self.sliding_window = sliding_window
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
            LoadImaged(keys="image", dtype=np.uint8),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys="image"),
        ]

    def inferer(self):
        return (
            SlidingWindowInferer(
                roi_size=(2048, 2048),
                sw_batch_size=4,
                overlap=0,
                mode=BlendMode.GAUSSIAN,
            )
            if self.sliding_window
            else SimpleInferer()
        )

    def post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            SqueezeDimd(keys="pred"),
            Transposed(keys="pred", indices=(1, 0)),
            ToNumpyd(keys="pred", dtype=np.uint8),
        ]
