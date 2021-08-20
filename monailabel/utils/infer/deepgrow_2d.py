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

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    Fetch2DSliced,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropGuidanced,
)
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsChannelLastd,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    Spacingd,
    ToNumpyd,
)

from monailabel.interfaces.tasks import InferTask, InferType


class InferDeepgrow2D(InferTask):
    """
    This provides Inference Engine for Deepgrow-2D pre-trained model.
    For More Details, Refer https://github.com/Project-MONAI/tutorials/tree/master/deepgrow/ignite
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPGROW,
        labels=None,
        dimension=2,
        description="A pre-trained 2D DeepGrow model based on UNET",
        spatial_size=(256, 256),
        model_size=(256, 256, 256),
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

        self.spatial_size = spatial_size
        self.model_size = model_size

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            AsChannelFirstd(keys="image"),
            Spacingd(keys="image", pixdim=[1.0, 1.0], mode="bilinear"),
            AddGuidanceFromPointsd(ref_image="image", guidance="guidance", dimensions=2),
            Fetch2DSliced(keys="image", guidance="guidance"),
            AddChanneld(keys="image"),
            SpatialCropGuidanced(keys="image", guidance="guidance", spatial_size=[256, 256]),
            Resized(keys="image", spatial_size=[256, 256], mode="area"),
            ResizeGuidanced(guidance="guidance", ref_image="image"),
            NormalizeIntensityd(keys="image", subtrahend=208, divisor=388),
            AddGuidanceSignald(image="image", guidance="guidance"),
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ToNumpyd(keys="pred"),
            RestoreLabeld(keys="pred", ref_image="image", mode="nearest"),
            AsChannelLastd(keys="pred"),
        ]
