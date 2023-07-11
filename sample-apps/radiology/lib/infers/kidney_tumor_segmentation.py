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

from lib.transforms.transforms import KidneyLabels
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityRanged,
    GaussianSmoothd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class KidneyTumorSeg(BasicInferTask):
    """
    This provides Inference Engine for pre-trained Segmentation (SegResNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) Segmentation from CT image",
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
        if data and isinstance(data.get("image"), str):
            t = [
                LoadImaged(keys="image"),
                EnsureTyped(keys="image", device=data.get("device") if data else None),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), allow_missing_keys=True),
                ScaleIntensityRanged(keys="image", a_min=-1000, a_max=600, b_min=0.0, b_max=1.0, clip=True),
                GaussianSmoothd(keys="image", sigma=0.2),
            ]
        else:
            t = [
                EnsureChannelFirstd(keys="label"),
                KidneyLabels(keys="label"),
                Orientationd(keys=("image", "label"), axcodes="RAS"),
                Spacingd(keys=("image", "label"), pixdim=self.target_spacing, allow_missing_keys=True),
                # NormalizeIntensityd(keys="image", nonzero=True),
                ScaleIntensityRanged(keys="image", a_min=-1000, a_max=600, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys="image", source_key="label", margin=[100, 10, 50]), # Direction -> Sagittal, Coronal, Axial
                GaussianSmoothd(keys="image", sigma=0.2),
            ]
        return t

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=2,
            overlap=0.4,
            padding_mode="replicate",
            mode="gaussian",
        )

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ]

        if data and data.get("largest_cc", False):
            t.append(KeepLargestConnectedComponentd(keys="pred"))
        t.extend([
            Restored(keys="pred", ref_image="image")
        ])
        return t
