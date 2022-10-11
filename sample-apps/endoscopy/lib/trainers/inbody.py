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

import numpy as np
import torch
from ignite.metrics import Accuracy
from lib.transforms import LabelToBinaryClassd
from monai.handlers import from_engine
from monai.inferers import SimpleInferer
from monai.transforms import (
    AsChannelFirstd,
    AsDiscreted,
    CastToTyped,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ToTensord,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class InBody(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        description="Endoscopy Classification for InBody/OutBody",
        **kwargs,
    ):
        self._network = network
        self.labels = labels
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.001)

    def loss_function(self, context: Context):
        return torch.nn.CrossEntropyLoss(reduction="sum")

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            LabelToBinaryClassd(keys="label", offset=2),
            ToTensord(keys=("image", "label")),
            AsChannelFirstd("image"),
            Resized(keys="image", spatial_size=(256, 256), mode="bilinear"),
            CastToTyped(keys="image", dtype=torch.float32),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys="image"),
            RandRotated(keys="image", range_x=0.3, prob=0.5, mode="bilinear"),
            RandScaleIntensityd(keys="image", factors=0.3, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", std=0.01, prob=0.5),
            RandFlipd(keys="image", spatial_axis=0, prob=0.5),
            RandFlipd(keys="image", spatial_axis=1, prob=0.5),
        ]

    def train_post_transforms(self, context: Context):
        return [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=(2, 2))]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            LabelToBinaryClassd(keys="label", offset=2),
            ToTensord(keys=("image", "label")),
            AsChannelFirstd("image"),
            Resized(keys="image", spatial_size=(256, 256), mode="bilinear"),
            CastToTyped(keys="image", dtype=torch.float32),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys="image"),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_key_metric(self, context: Context):
        return {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_key_metric(self, context):
        return {"val_mean_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}
