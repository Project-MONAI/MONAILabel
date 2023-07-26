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
from typing import Callable, Sequence

from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    ToNumpyd,
    CropForegroundd,
    Spacingd,
    KeepLargestConnectedComponentd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored
from lib.transforms.transforms_brats import NRRDWriterBrain


class VascSegmentation(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model
    """

    def __init__(
        self,
        path,
        network=None,
        spatial_size=(128, 128, 128),
        target_spacing=(0.5, 0.5, 0.8),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation over 3D Images",
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
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        # This should be work once we train a model for arteries and veins using all 4 modalities
        if data and isinstance(data.get("image"), str):
            t = [
                LoadImaged(keys="image", reader='itkreader'),
                EnsureChannelFirstd(keys="image"),
                Spacingd(keys="image", pixdim=self.target_spacing),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        else:
            t = [
                 # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                 EnsureChannelFirstd(keys="label"),
                 CropForegroundd(keys="image", source_key="label", margin=20, allow_missing_keys=True),
                 ]

        t.extend(
            [
                EnsureTyped(keys="image"),
            ]
        )
        return t

        # return [
        #     LoadImaged(keys="image", reader='itkreader'),
        #     EnsureChannelFirstd(keys="image"),
        #     NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        #     CropForegroundd(keys="image", source_key="label", margin=10, allow_missing_keys=True),
        #     # ScaleIntensityRanged(keys="image", a_min=800, a_max=5000, b_min=0.0, b_max=1.0, clip=True),
        #     EnsureTyped(keys="image"),
        # ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=self.spatial_size, sw_batch_size=4, overlap=0.5, padding_mode="replicate", mode="gaussian"
        )

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred", num_components=2),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
        return t

    def writer(self, data, extension=None, dtype=None):
        writer = NRRDWriterBrain(label="pred", original_label_indexing=self.labels)
        return writer(data)
