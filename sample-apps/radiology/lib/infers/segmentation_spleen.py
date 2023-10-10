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

from typing import Callable, Sequence

from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class SegmentationSpleen(BasicInferTask):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=self.target_spacing, allow_missing_keys=True),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=[160, 160, 160],
            sw_batch_size=1,
            overlap=0.25,
        )

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
