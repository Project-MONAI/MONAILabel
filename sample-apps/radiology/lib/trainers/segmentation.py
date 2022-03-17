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
import logging

import torch
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics
from monailabel.transform.pre import FindAllValidSlicesByClassd, RandomForegroundCropSamplesd

logger = logging.getLogger(__name__)


class Segmentation(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        spatial_size=(128, 128, 128),
        num_samples=8,
        description="Train Segmentation model",
        **kwargs,
    ):
        self._network = network
        self.spatial_size = spatial_size
        self.num_samples = num_samples
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.AdamW(context.network.parameters(), amsgrad=True)

    def loss_function(self, context: Context):
        return DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            AddChanneld(keys=("image", "label")),
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
                align_corners=(True, True),
            ),
            ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            FindAllValidSlicesByClassd(label="label", sids="sids"),
            RandomForegroundCropSamplesd(
                keys=("image", "label"),
                label_key="label",
                sids_key="sids",
                num_samples=self.num_samples,
                spatial_size=self.spatial_size,
            ),
            SpatialPadd(keys=("image", "label"), spatial_size=self.spatial_size),
            EnsureTyped(keys=("image", "label"), device=context.device),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
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
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            AddChanneld(keys=("image", "label")),
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
                align_corners=(True, True),
            ),
            ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=("image", "label"), device=context.device),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=(160, 160, 160))

    def train_key_metric(self, context: Context):
        return region_wise_metrics(self._labels, self.TRAIN_KEY_METRIC, "train")

    def val_key_metric(self, context: Context):
        return region_wise_metrics(self._labels, self.VAL_KEY_METRIC, "val")

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
