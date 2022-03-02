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
from typing import Any, Callable, Dict, Sequence

import numpy as np
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    EnsureTyped,
    ScaleIntensityRangeD,
    SqueezeDimd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.writer import PolygonWriter

from .transforms import FilterImaged, FindContoursd, LoadImagePatchd, PostFilterLabeld

logger = logging.getLogger(__name__)


class InferSegmentation(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(256, 256),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained semantic segmentation model for Nuclei (Pathology)",
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

    def info(self) -> Dict[str, Any]:
        i = super().info()
        i["pathology"] = True
        return i

    def config(self) -> Dict[str, Any]:
        c = super().config()
        c.update(
            {
                "roi_size": self.roi_size,
                "sw_batch_size": 2,
            }
        )
        return c

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImagePatchd(keys="image", conversion="RGB", dtype=np.uint8),
            FilterImaged(keys="image"),
            AsChannelFirstd(keys="image"),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            EnsureTyped(keys="image"),
        ]

    def inferer(self, data=None) -> Callable:
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

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys="pred",
                argmax=True,
            ),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys=("image", "pred")),
            PostFilterLabeld(keys="pred", image="image"),
            FindContoursd(keys="pred", labels=self.labels),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
