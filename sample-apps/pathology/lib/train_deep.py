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
from monai.apps.deepgrow.transforms import AddGuidanceSignald, AddRandomGuidanced, FindDiscrepancyRegionsd
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
    RandRotate90d,
    RandSpatialCropSamplesd,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
)

from monailabel.deepedit.interaction import Interaction
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from .handlers import TensorBoardImageHandler
from .transforms import AddInitialSeedPointExd, ClipBorderd, FilterImaged, LabelToChanneld, NormalizeImaged

logger = logging.getLogger(__name__)


class MyDeepgrow(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        max_train_interactions=10,
        max_val_interactions=5,
        patch_size=(512, 512),
        num_samples=16,
        description="Pathology Interactive Segmentation (BCSS Dataset)",
        **kwargs,
    ):
        self._network = network
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.labels = labels
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(self._network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def get_click_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanced(guidance="guidance", discrepancy="discrepancy", probability="probability"),
            AddGuidanceSignald(image="image", guidance="guidance", number_intensity_ch=3),
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            FilterImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            LabelToChanneld(keys="label", labels=self.labels),
            ClipBorderd(keys=("image", "label"), border=40),
            BorderPadd(keys=("image", "label"), spatial_border=40),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            RandSpatialCropSamplesd(
                keys=("image", "label"),
                roi_size=self.patch_size,
                random_center=True,
                random_size=False,
                num_samples=self.num_samples,
            ),
            RandRotate90d(keys=("image", "label"), prob=0.5, spatial_axes=(0, 1)),
            NormalizeImaged(keys="image"),
            AddInitialSeedPointExd(label="label", guidance="guidance"),
            AddGuidanceSignald(image="image", guidance="guidance", number_intensity_ch=3),
            EnsureTyped(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
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
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir, batch_limit=0))
        return handlers

    def _dataloader(self, context, dataset, batch_size, num_workers):
        return DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=list_data_collate
        )

    def train_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=0.5,
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_train_interactions,
            train=True,
        )

    def val_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=1.0,
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_val_interactions,
            train=False,
        )
