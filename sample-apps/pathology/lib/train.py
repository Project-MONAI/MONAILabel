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

import numpy as np
import torch
from ignite.metrics import Accuracy
from monai.data import DataLoader, list_data_collate
from monai.handlers import from_engine
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    BorderPadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from .handlers import TensorBoardImageHandler
from .transforms import LabelToChanneld

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        patch_size=(512, 512),
        num_samples=16,
        description="Pathology Semantic Segmentation (BCSS Dataset)",
        **kwargs,
    ):
        self._network = network
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.labels = labels
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(self._network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(sigmoid=True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            EnsureChannelFirstd(keys="image"),
            LabelToChanneld(keys="label", labels=self.labels),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            ScaleIntensityd(keys=("image", "label")),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=(self.patch_size[0] - 4, self.patch_size[1] - 4),
                pos=1,
                neg=1,
                num_samples=self.num_samples,
            ),
            BorderPadd(keys=("image", "label"), spatial_border=2),
            RandRotate90d(keys=("image", "label"), prob=0.5, spatial_axes=(0, 1)),
            EnsureTyped(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]

    def train_key_metric(self, context: Context):
        return {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_key_metric(self, context: Context):
        return {"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=self.patch_size, sw_batch_size=4)

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir))
        return handlers

    def _dataloader(self, context, dataset, batch_size, num_workers):
        return DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=list_data_collate
        )
