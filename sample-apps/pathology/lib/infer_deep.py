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
from monai.apps.deepgrow.transforms import AddGuidanceSignald
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import Activationsd, AsDiscreted, EnsureChannelFirstd, EnsureTyped, SqueezeDimd, ToNumpyd

from monailabel.interfaces.tasks.infer import InferTask, InferType

from .transforms import (
    AddClickGuidanced,
    FilterImaged,
    FindContoursd,
    LoadImagePatchd,
    NormalizeImaged,
    PostFilterLabeld,
)

logger = logging.getLogger(__name__)


class InferDeep(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(1024, 1024),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained semantic segmentation model for Tumor (Pathology)",
    ):
        self.roi_size = roi_size
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def config(self):
        c = super().config()
        c.update(
            {
                "roi_size": self.roi_size,
                "sw_batch_size": 2,
            }
        )
        return c

    def pre_transforms(self, data=None):
        return [
            LoadImagePatchd(keys="image", conversion="RGB", dtype=np.uint8),
            FilterImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            NormalizeImaged(keys="image"),
            AddClickGuidanced(image="image", guidance="guidance"),
            AddGuidanceSignald(image="image", guidance="guidance", number_intensity_ch=3),
            EnsureTyped(keys="image"),
        ]

    def inferer(self, data=None):
        roi_size = data.get("roi_size", self.roi_size) if data else self.roi_size
        input_shape = data["image"].shape if data else None
        sw_batch_size = data.get("sw_batch_size", 2) if data else 2
        device = data.get("device")

        if input_shape and (input_shape[-1] > roi_size[-1] or input_shape[-2] > roi_size[-2]):
            return SlidingWindowInferer(
                roi_size=data.get("roi_size", roi_size),
                sw_batch_size=data.get("sw_batch_size", sw_batch_size),
                sw_device=device,
                device=device,
            )
        return SimpleInferer()

    def post_transforms(self, data=None):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            SqueezeDimd(keys="pred"),
            ToNumpyd(keys=("image", "pred")),
            PostFilterLabeld(keys="pred", image="image"),
            FindContoursd(keys="pred"),
        ]
