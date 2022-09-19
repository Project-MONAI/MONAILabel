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
from lib.transforms.transforms import (
    ConcatenateROId,
    GaussianSmoothedCentroidd,
    GetCentroidsd,
    SelectVertebraAndCroppingd,
)
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    GaussianSmoothd,
    LoadImaged,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRanged,
    SelectItemsd,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics

logger = logging.getLogger(__name__)


class SegmentationVertebra(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        roi_size=(96, 96, 96),
        target_spacing=(1.0, 1.0, 1.0),
        num_samples=4,
        description="Train vertebra segmentation model",
        **kwargs,
    ):
        self._network = network
        self.roi_size = roi_size
        self.target_spacing = target_spacing
        self.num_samples = num_samples
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.AdamW(context.network.parameters(), lr=1e-4, weight_decay=1e-5)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def lr_scheduler_handler(self, context: Context):
        return None

    def train_data_loader(self, context, num_workers=0, shuffle=False):
        return super().train_data_loader(context, num_workers, True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            # NormalizeIntensityd(keys="image", divisor=2048.0),
            ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1900, b_min=0.0, b_max=1.0, clip=True),
            GaussianSmoothd(keys="image", sigma=0.4),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            GetCentroidsd(keys="label", centroids_key="centroids"),
            SelectVertebraAndCroppingd(keys="label"),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            GaussianSmoothedCentroidd(keys="NA", signal_key="signal"),
            Resized(keys=("image", "label", "signal"), spatial_size=self.roi_size, mode=("area", "nearest", "area")),
            # SaveImaged(keys="label", output_postfix="", output_dir="/home/andres/Downloads", separate_folder=False),
            ConcatenateROId(keys="signal"),
            EnsureTyped(keys=("image", "label"), device=context.device),
            SelectItemsd(keys=("image", "label", "centroids", "original_size", "current_label", "slices_cropped")),
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
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            # NormalizeIntensityd(keys="image", divisor=2048.0),
            ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1900, b_min=0.0, b_max=1.0, clip=True),
            GaussianSmoothd(keys="image", sigma=0.4),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            GetCentroidsd(keys="label", centroids_key="centroids"),
            SelectVertebraAndCroppingd(keys="label"),
            GaussianSmoothedCentroidd(keys="NA", signal_key="signal"),
            Resized(keys=("image", "label", "signal"), spatial_size=self.roi_size, mode=("area", "nearest", "area")),
            ConcatenateROId(keys="signal"),
            EnsureTyped(keys=("image", "label"), device=context.device),
            SelectItemsd(keys=("image", "label", "centroids", "original_size", "current_label", "slices_cropped")),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

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
