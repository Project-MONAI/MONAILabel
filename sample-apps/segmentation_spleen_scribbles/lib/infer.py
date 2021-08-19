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

from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CopyItemsd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    ToNumpyd,
)

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import BoundingBoxd, Restored

from .transforms import WriteLogits


class SegmentationWithWriteLogits(InferTask):
    """
    Inference Engine for pre-trained Spleen segmentation (UNet) model for MSD Dataset. It additionally provides
    appropriate transforms to save logits that are needed for post processing stage.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="spleen",
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
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
            AddChanneld(keys="image"),
            Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0]),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=[160, 160, 160])

    def post_transforms(self):
        return [
            CopyItemsd(keys="pred", times=1, names="logits"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys=["pred", "logits"]),
            Restored(keys=["pred", "logits"], ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
            WriteLogits(key="logits", result="result"),
        ]
