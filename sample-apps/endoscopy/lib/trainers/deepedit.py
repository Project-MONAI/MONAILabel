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
from lib.trainers.tooltracking import MeanIoUMetric
from monai.apps.deepgrow.transforms import AddRandomGuidanced, FindDiscrepancyRegionsd
from monai.handlers import MeanDice, from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityRangeD,
    SelectItemsd,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
)

from monailabel.deepedit.handlers import TensorBoard2DImageHandler
from monailabel.deepedit.interaction import Interaction
from monailabel.deepedit.transforms import AddGuidanceSignald, AddInitialSeedPointd
from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.transform.pre import NormalizeLabeld

logger = logging.getLogger(__name__)


class DeepEdit(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        dimension,
        roi_size,
        max_train_interactions,
        max_val_interactions,
        description="Train DeepEdit Model for Endoscopy",
        **kwargs,
    ):
        self._network = network
        self.dimension = dimension
        self.roi_size = roi_size
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions

        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def get_click_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanced(guidance="guidance", discrepancy="discrepancy", probability="probability"),
            AddGuidanceSignald(keys="image", guidance="guidance"),
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            EnsureChannelFirstd(keys="image"),
            AddChanneld(keys="label"),
            Resized(keys=("image", "label"), spatial_size=self.roi_size, mode=("area", "nearest")),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            RandRotated(keys=("image", "label"), range_x=np.pi, prob=0.5, mode=("bilinear", "nearest")),
            RandZoomd(keys=("image", "label"), min_zoom=0.8, max_zoom=1.2, prob=0.5, mode=("bilinear", "nearest")),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            NormalizeLabeld(keys="label", value=1),
            AddInitialSeedPointd(keys="guidance", label="label", connected_regions=5),
            AddGuidanceSignald(keys="image", guidance="guidance"),
            SelectItemsd(keys=("image", "label", "guidance")),
            EnsureTyped(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def val_key_metric(self, context):
        return {
            "val_mean_iou": MeanIoUMetric(output_transform=from_engine(["pred", "label"])),
            self.VAL_KEY_METRIC: MeanDice(output_transform=from_engine(["pred", "label"])),
        }

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoard2DImageHandler(log_dir=context.events_dir, batch_limit=0))
        return handlers

    def val_handlers(self, context: Context):
        handlers = super().val_handlers(context)
        if context.local_rank == 0 and handlers:
            handlers.append(TensorBoard2DImageHandler(log_dir=context.events_dir, batch_limit=0, tag_prefix="val-"))
        return handlers

    def train_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=0.5,
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_train_interactions,
            train=True,
        )

    def val_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=0.5,
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_val_interactions,
            train=False,
        )
