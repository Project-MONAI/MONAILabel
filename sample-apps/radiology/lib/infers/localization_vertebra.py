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

import torch
from lib.transforms.transforms import VertebraLocalizationPostProcessing
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    GaussianSmoothd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored


class SimpleJsonWriter:
    def __init__(self, label="pred"):
        self.label = label

    def __call__(self, data):
        return None, data["result"]


class LocalizationVertebra(InferTask):
    """
    This provides Inference Engine for pre-trained vertebra localization (UNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) vertebra localization from CT image",
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
            LoadImaged(keys=("image", "first_stage_pred"), reader="ITKReader"),
            EnsureTyped(keys=("image", "first_stage_pred"), device=data.get("device") if data else None),
            EnsureChannelFirstd(keys=("image", "first_stage_pred")),
            Orientationd(keys=("image", "first_stage_pred"), axcodes="RAS"),
            Spacingd(keys=("image", "first_stage_pred"), pixdim=self.target_spacing, mode="bilinear"),
            CropForegroundd(keys=("image", "first_stage_pred"), source_key="image"),
            GaussianSmoothd(keys="image", sigma=0.75),
            NormalizeIntensityd(keys="image", divisor=2048.0),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            SpatialPadd(keys="image", spatial_size=self.roi_size),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(roi_size=self.roi_size)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", other=torch.nn.functional.leaky_relu),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
            ScaleIntensityd(keys="pred", minv=0.0, maxv=100.0),
            VertebraLocalizationPostProcessing(keys="pred", result="result"),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = SimpleJsonWriter(label=self.output_label_key)
        return writer(data)
