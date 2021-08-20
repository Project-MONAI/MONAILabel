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
from monai.apps.deepgrow.interaction import Interaction
from monai.apps.deepgrow.transforms import (
    AddGuidanceSignald,
    AddInitialSeedPointd,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandRotated,
    Resized,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.custom_tensorboard_handlers import TensorBoardImageHandler
from monailabel.deepedit.transforms import ClickRatioAddRandomGuidanced, DiscardAddGuidanced
from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train DeepEdit model for spleen over 3D CT Images",
        model_size=(256, 256, 128),
        max_train_interactions=20,
        max_val_interactions=10,
        **kwargs,
    ):
        self._network = network
        self.model_size = model_size
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions

        super().__init__(model_dir, description, **kwargs)

    def network(self):
        return self._network

    def optimizer(self):
        return torch.optim.Adam(self._network.parameters(), lr=0.0001)

    def loss_function(self):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def get_click_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            ClickRatioAddRandomGuidanced(guidance="guidance", discrepancy="discrepancy", probability="probability"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            DiscardAddGuidanced(image="image", probability=0.5),
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            RandAdjustContrastd(keys="image", gamma=6),
            RandHistogramShiftd(keys="image", num_control_points=8, prob=0.5),
            RandRotated(
                keys=("image", "label"),
                range_x=0.3,
                range_y=0.3,
                range_z=0.3,
                prob=0.4,
                keep_size=True,
                mode=("bilinear", "nearest"),
            ),
            Resized(keys=("image", "label"), spatial_size=self.model_size, mode=("area", "nearest")),
            FindAllValidSlicesd(label="label", sids="sids"),
            AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            DiscardAddGuidanced(image="image", probability=0.5),
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ]

    def val_pre_transforms(self):
        return self.train_pre_transforms()

    def val_inferer(self):
        return SimpleInferer()

    def train_iteration_update(self):
        return Interaction(
            transforms=self.get_click_transforms(),
            max_interactions=self.max_train_interactions,
            key_probability="probability",
            train=True,
        )

    def val_iteration_update(self):
        return Interaction(
            transforms=self.get_click_transforms(),
            max_interactions=self.max_val_interactions,
            key_probability="probability",
            train=False,
        )

    def train_handlers(self):
        handlers = super().train_handlers()
        handlers.append(TensorBoardImageHandler(log_dir=self.events_dir, epoch_level=True))
        return handlers
