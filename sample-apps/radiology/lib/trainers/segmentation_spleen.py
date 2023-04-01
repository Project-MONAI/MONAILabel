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

import logging

import torch
from monai.apps.deepedit.transforms import NormalizeLabelsInDatasetd
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    SelectItemsd,
    Spacingd,
    ToTensord,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class SegmentationSpleen(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        roi_size=(96, 96, 96),
        target_spacing=(1.0, 1.0, 1.0),
        description="Train Segmentation model for spleen",
        **kwargs,
    ):
        self._network = network
        self.roi_size = roi_size
        self.target_spacing = target_spacing
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.AdamW(context.network.parameters(), lr=1e-4, weight_decay=1e-5)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, batch=True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            EnsureChannelFirstd(keys=("image", "label")),
            EnsureTyped(keys=("image", "label"), device=context.device),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
                random_size=False,
            ),
            SelectItemsd(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            ToTensord(keys=("pred", "label")),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=2,
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            EnsureTyped(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            SelectItemsd(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=[160, 160, 160], sw_batch_size=1, overlap=0.25)
