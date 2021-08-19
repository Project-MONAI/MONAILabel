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

import copy
import logging

import torch
from monai.apps.deepgrow.interaction import Interaction
from monai.apps.deepgrow.transforms import (
    AddGuidanceSignald,
    AddInitialSeedPointd,
    AddRandomGuidanced,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
    SpatialCropForegroundd,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

from .transforms import Random2DSlice

logger = logging.getLogger(__name__)


class TrainDeepgrow(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        dimension,
        roi_size,
        model_size,
        max_train_interactions,
        max_val_interactions,
        description="Train Deepgrow Model",
        **kwargs,
    ):
        self._network = network
        self.dimension = dimension
        self.roi_size = roi_size
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

    def partition_datalist(self, request, datalist, shuffle=True):
        train_ds, val_ds = super().partition_datalist(request, datalist, shuffle)
        if self.dimension != 2:
            return train_ds, val_ds

        flatten_train_ds = []
        for _ in range(max(request.get("train_random_slices", 20), 1)):
            flatten_train_ds.extend(copy.deepcopy(train_ds))
        logger.info(f"After flatten:: {len(train_ds)} => {len(flatten_train_ds)}")

        flatten_val_ds = []
        for _ in range(max(request.get("val_random_slices", 5), 1)):
            flatten_val_ds.extend(copy.deepcopy(val_ds))
        logger.info(f"After flatten:: {len(val_ds)} => {len(flatten_val_ds)}")
        return flatten_train_ds, flatten_val_ds

    def get_click_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanced(guidance="guidance", discrepancy="discrepancy", probability="probability"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self):
        # Dataset preparation
        t = [
            LoadImaged(keys=("image", "label")),
            AsChannelFirstd(keys=("image", "label")),
            Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
        ]

        # Pick random slice (run more epochs to cover max slices for 2D training)
        if self.dimension == 2:
            t.append(Random2DSlice(image="image", label="label"))

        # Training
        t.extend(
            [
                AddChanneld(keys=("image", "label")),
                SpatialCropForegroundd(keys=("image", "label"), source_key="label", spatial_size=self.roi_size),
                Resized(keys=("image", "label"), spatial_size=self.model_size, mode=("area", "nearest")),
                NormalizeIntensityd(keys="image"),
            ]
        )
        if self.dimension == 3:
            t.append(FindAllValidSlicesd(label="label", sids="sids"))
        t.extend(
            [
                AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
                AddGuidanceSignald(image="image", guidance="guidance"),
                ToTensord(keys=("image", "label")),
            ]
        )

        return t

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
