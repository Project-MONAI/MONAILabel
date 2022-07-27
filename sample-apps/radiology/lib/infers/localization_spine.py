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
    GaussianSmoothd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored


class LocalizationSpine(InferTask):
    """
    This provides Inference Engine for pre-trained spine localization (UNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) spine localization from CT image",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=self.target_spacing),
            GaussianSmoothd(keys="image", sigma=0.75),
            NormalizeIntensityd(keys="image", divisor=2048.0),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(roi_size=self.roi_size)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True), AsDiscreted(keys="pred", argmax=True), ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image")
        ]
