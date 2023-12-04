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
from lib.transforms.transforms import NormalizeLabelsInDatasetd
from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    GaussianSmoothd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityd,
    SelectItemsd,
    Spacingd, ScaleIntensityRanged, RandAffined, RandGaussianSmoothd, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics

logger = logging.getLogger(__name__)


class Segmentation(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        roi_size=(96, 96, 96),
        target_spacing=(1.0, 1.0, 1.0),
        num_samples=4,
        description="Train Segmentation model",
        **kwargs,
    ):
        self._network = network
        self.roi_size = roi_size
        self.target_spacing = target_spacing
        self.num_samples = num_samples
        super().__init__(model_dir, description, val_interval=100, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.AdamW(context.network.parameters(), lr=2e-4, weight_decay=1e-5)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def lr_scheduler_handler(self, context: Context):
        return None

    def train_data_loader(self, context, num_workers=0, shuffle=False):
        return super().train_data_loader(context, num_workers, True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader='itkreader'),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            EnsureChannelFirstd(keys=("image", "label")),
            EnsureTyped(keys=("image", "label"), device=context.device),
            CropForegroundd(
                keys=("image", "label"),
                source_key="image",
                margin=10,
                k_divisible=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
            ),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            EnsureSameShaped(
                keys="label", source_key="image", allow_missing_keys=True
            ),
            ScaleIntensityRanged(keys="image", a_min=1.1, a_max=103, b_min=-1, b_max=1, clip=True),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
                random_size=False,
            ),
            RandAffined(
                keys=("image", "label"),
                prob=0.2,
                rotate_range=[0.26, 0.26, 0.26],
                scale_range=[0.2, 0.2, 0.2],
                mode=["bilinear", "nearest"],
                spatial_size=self.roi_size,
                cache_grid=True,
                padding_mode="border",
            ),
            RandGaussianSmoothd(keys="image", prob=0.2, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]),
            RandScaleIntensityd(keys="image", prob=0.5, factors=0.3),
            RandShiftIntensityd(keys="image", prob=0.5, offsets=0.1),
            RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
            SelectItemsd(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            EnsureTyped(keys="pred", device=context.device),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=len(self._labels) + 1,
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader='itkreader'),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            EnsureTyped(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            CropForegroundd(
                keys=("image", "label"),
                source_key="image",
                margin=10,
                k_divisible=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
            ),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            EnsureSameShaped(
                keys="label", source_key="image", allow_missing_keys=True
            ),
            ScaleIntensityRanged(keys="image", a_min=1.1, a_max=103, b_min=-1, b_max=1, clip=True),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
                random_size=False,
            ),
            SelectItemsd(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=1,
            overlap=0.625,
            mode="gaussian",
            cache_roi_weight_map=True,
            progress=False
        )

    def norm_labels(self):
        # This should be applied along with NormalizeLabelsInDatasetd transform
        new_label_nums = {}
        for idx, (key_label, val_label) in enumerate(self._labels.items(), start=1):
            if key_label != "background":
                new_label_nums[key_label] = idx
            if key_label == "background":
                new_label_nums["background"] = 0
        return new_label_nums

    def train_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "train_mean_dice", "train")

    def val_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "val_mean_dice", "val")

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(
                TensorBoardImageHandler(
                    log_dir=context.events_dir,
                    batch_transform=from_engine(["image", "label"]),
                    output_transform=from_engine(["pred"]),
                    interval=20,
                    epoch_level=True,
                )
            )
        return handlers
