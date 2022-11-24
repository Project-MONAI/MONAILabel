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
import os

import numpy as np
import torch
from ignite.metrics import Accuracy
from lib.handlers import TensorBoardImageHandler
from lib.utils import split_dataset
from monai.handlers import from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandTorchVisiond,
    ScaleIntensityRangeD,
    ScaleIntensityRanged,
)

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class SegmentationNuclei(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        tile_size=(512, 512),
        patch_size=256,
        description="Pathology Semantic Segmentation for Nuclei",
        **kwargs,
    ):
        self._network = network
        self.tile_size = tile_size
        self.patch_size = patch_size
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def pre_process(self, request, datastore: Datastore):
        self.cleanup(request)

        cache_dir = os.path.join(self.get_cache_dir(request), "train_ds")
        source = request.get("dataset_source")
        max_region = request.get("dataset_max_region", (10240, 10240))
        max_region = (max_region, max_region) if isinstance(max_region, int) else max_region[:2]

        return split_dataset(
            datastore=datastore,
            cache_dir=cache_dir,
            source=source,
            groups=self._labels,
            tile_size=self.tile_size,
            max_region=max_region,
            limit=request.get("dataset_limit", 0),
            randomize=request.get("dataset_randomize", True),
            crop_size=self.patch_size,
        )

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            EnsureTyped(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            RandTorchVisiond(
                keys="image",
                name="ColorJitter",
                # prob=0.5,
                brightness=64.0 / 255.0,
                contrast=0.75,
                saturation=0.25,
                hue=0.04,
            ),
            RandFlipd(keys=("image", "label"), prob=0.5),
            RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3, spatial_axes=(0, 1)),
            ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ]

    def train_post_transforms(self, context: Context):
        return [
            EnsureTyped(keys="pred", device=context.device),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=len(self._labels) + 1),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            EnsureTyped(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ]

    def train_additional_metrics(self, context: Context):
        return {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_additional_metrics(self, context: Context):
        return {"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir, batch_limit=4, tag_name="train"))
        return handlers

    def val_handlers(self, context: Context):
        handlers = super().val_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir, batch_limit=8, tag_name="val"))
        return handlers
