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

from typing import Callable, Sequence, Union

from lib.transforms.transforms import ConcatenateROId, CropAndCreateSignald, PlaceCroppedAread
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
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureChannelFirstd(keys="image"),
            # NormalizeIntensityd(keys="image", divisor=2048.0),
            # to make sure the target size is same as train
            Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),
            ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1900, b_min=0.0, b_max=1.0, clip=True),
            GaussianSmoothd(keys="image", sigma=0.4),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            CropAndCreateSignald(keys="image", signal_key="signal"),
            # Make sure the image and signal have the same size as roi
            Resized(keys=("image", "signal"), spatial_size=self.roi_size, mode=("area", "area")),
            ConcatenateROId(keys="signal"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    # SHOULD WE DO THIS? HOW TO RESIZE THE PREDICTION?
    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        largest_cc = False if not data else data.get("largest_cc", False)
        applied_labels = list(self.labels.values()) if isinstance(self.labels, dict) else self.labels
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ]
        if largest_cc:
            t.append(KeepLargestConnectedComponentd(keys="pred", applied_labels=applied_labels))
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
