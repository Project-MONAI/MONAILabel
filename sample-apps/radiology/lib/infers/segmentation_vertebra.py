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

from lib.transforms.transforms import ConcatenateROId, CropAndCreateSignald, GetOriginalInformation, PlaceCroppedAread
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    GaussianSmoothd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored


class SegmentationVertebra(InferTask):
    """
    This provides Inference Engine for pre-trained vertebra segmentation (UNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) vertebra segmentation from CT image",
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
                LoadImaged(keys="image", reader="ITKReader"),
                EnsureChannelFirstd(keys="image"),
                GetOriginalInformation(keys="image"),
                Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),
                ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1900, b_min=0.0, b_max=1.0, clip=True),
                GaussianSmoothd(keys="image", sigma=0.4),
                ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            ]
        else:
            t = []

        t.extend(
            [
                CropAndCreateSignald(keys="image", signal_key="signal"),
                Resized(keys=("image", "signal"), spatial_size=self.roi_size, mode=("area", "area")),
                ConcatenateROId(keys="signal"),
            ]
        )
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        applied_labels = list(self.labels.values()) if isinstance(self.labels, dict) else self.labels
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=applied_labels),
        ]

        if not data or not data.get("pipeline_mode", False):
            t.extend(
                [
                    ToNumpyd(keys="pred"),
                    PlaceCroppedAread(keys="pred"),
                    Restored(keys="pred", ref_image="image"),
                ]
            )
        return t

    def is_valid(self) -> bool:
        return False

    def writer(self, data, extension=None, dtype=None):
        if data.get("pipeline_mode", False):
            return {
                "image": data["image"],
                "pred": data["pred"],
                "slices_cropped": data["slices_cropped"],
                "cropped_size": data["cropped_size"],
                "current_label" : data["current_label"],
            }, {}

        return super().writer(data, extension, dtype)
