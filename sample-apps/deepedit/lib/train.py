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

from monailabel.deepedit.handlers import TensorBoardImageHandler
from monailabel.deepedit.interaction import Interaction
from monailabel.deepedit.transforms import PosNegClickProbAddRandomGuidanced, SingleLabelSingleModalityd
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train DeepEdit model for 3D Images",
        spatial_size=(128, 128, 64),
        target_spacing=(1.0, 1.0, 1.0),
        deepgrow_probability_train=0.5,
        deepgrow_probability_val=1.0,
        max_train_interactions=20,
        max_val_interactions=10,
        debug_mode=False,
        **kwargs,
    ):
        self._network = network
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.deepgrow_probability_train = deepgrow_probability_train
        self.deepgrow_probability_val = deepgrow_probability_val
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions
        self.debug_mode = debug_mode

        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(self._network.parameters(), lr=0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def get_click_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            PosNegClickProbAddRandomGuidanced(
                guidance="guidance", discrepancy="discrepancy", probability="probability"
            ),
            AddGuidanceSignald(image="image", guidance="guidance"),
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="nibabelreader"),
            SingleLabelSingleModalityd(keys=("image", "label")),
            # RandZoomd(keys=("image", "label"), prob=0.4, min_zoom=0.3, max_zoom=1.9, mode=("bilinear", "nearest")),
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            RandAdjustContrastd(keys="image", gamma=6),
            RandHistogramShiftd(keys="image", num_control_points=8, prob=0.5),
            RandRotated(
                keys=("image", "label"),
                range_x=0.1,
                range_y=0.1,
                range_z=0.1,
                prob=0.4,
                keep_size=True,
                mode=("bilinear", "nearest"),
            ),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            FindAllValidSlicesd(label="label", sids="sids"),
            AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="nibabelreader"),
            SingleLabelSingleModalityd(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            FindAllValidSlicesd(label="label", sids="sids"),
            AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=self.deepgrow_probability_train,
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_train_interactions,
            click_probability_key="probability",
            train=True,
        )

    def val_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=self.deepgrow_probability_val,
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_val_interactions,
            click_probability_key="probability",
            train=False,
        )

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if self.debug_mode and context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir))
        return handlers
