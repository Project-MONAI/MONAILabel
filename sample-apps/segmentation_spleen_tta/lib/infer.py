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

from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import Restored


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
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
            EnsureChannelFirstd(keys="image"),
            Spacingd(
                keys="image",
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            Orientationd(keys="image", axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            Resized(keys="image", spatial_size=(128, 128, 128)),
            ToTensord(keys=["image"]),
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            ToTensord(keys=("image", "pred")),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
