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
import glob
import logging
import os

import torch
from monai.apps.deepedit.transforms import NormalizeLabelsInDatasetd
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandSpatialCropd,
    SelectItemsd,
    CropForegroundd,
    Spacingd,
    KeepLargestConnectedComponentd,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics

logger = logging.getLogger(__name__)


class VascSegmentation(BasicTrainTask):
    def __init__(
            self,
            model_dir,
            network,
            spatial_size=(128, 128, 128),  # Depends on original width, height and depth of the training images
            target_spacing=(0.5, 0.5, 0.8),
            num_samples=4,
            description="Train Segmentation model",
            **kwargs,
    ):
        self._network = network
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.num_samples = num_samples
        super().__init__(model_dir, description,  val_interval=50, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        # return Novograd(context.network.parameters(), 0.0001)
        return torch.optim.AdamW(context.network.parameters(), lr=1e-4, weight_decay=1e-5)

    def loss_function(self, context: Context):
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
        )

    def lr_scheduler_handler(self, context: Context):
        return None

    def train_data_loader(self, context, num_workers=0, shuffle=False):
        return super().train_data_loader(context, num_workers, True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader='itkreader'),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            # SaveImaged(keys="label", output_postfix="", output_dir="/home/andres/Downloads", separate_folder=False),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # ScaleIntensityRanged(keys="image", a_min=800, a_max=5000, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="label", margin=20, k_divisible=self.spatial_size[0]),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[self.spatial_size[0], self.spatial_size[1], self.spatial_size[2]],
                random_size=False,
            ),
            # RandScaleIntensityd(keys="image", factors=0.1, prob=0.8),
            # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.8),
            # RandGaussianSmoothd(keys="image", sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.8),
            EnsureTyped(keys=("image", "label"), device=context.device),
            SelectItemsd(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            EnsureTyped(keys="pred", device=context.device),
            Activationsd(keys="pred", softmax=len(self._labels) > 1, sigmoid=len(self._labels) == 1),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=(len(self._labels) + 1, len(self._labels) + 1),
            ),
            KeepLargestConnectedComponentd(keys="pred", num_components=2),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader='itkreader'),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # ScaleIntensityRanged(keys="image", a_min=500, a_max=5000, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="label", margin=20, k_divisible=self.spatial_size[0]),
            EnsureTyped(keys=("image", "label")),
            SelectItemsd(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(
            roi_size=self.spatial_size,
            sw_batch_size=1,
            overlap=0.25,
            # padding_mode="replicate",
            # mode="gaussian"
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

    # def partition_datalist(self, context: Context, shuffle=False):
    #     # Training images
    #     train_d = context.datalist
    #
    #     # Validation images
    #     data_dir = "/home/andres/Documents/workspace/arteries/"
    #     val_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nrrd")))
    #     val_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nrrd")))
    #     val_d = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]
    #
    #     if context.local_rank == 0:
    #         logger.info(f"Total Records for Training: {len(train_d)}")
    #         logger.info(f"Total Records for Validation: {len(val_d)}")
    #
    #     return train_d, val_d
