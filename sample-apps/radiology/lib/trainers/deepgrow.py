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
import pathlib
import shutil
from typing import Any, List

import torch
from monai.apps.deepgrow.dataset import create_dataset
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
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    SelectItemsd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class Deepgrow(BasicTrainTask):
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

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def pre_process(self, request, datastore: Datastore):
        self.cleanup(request)

        cache_dir = self.get_cache_dir(request)
        output_dir = os.path.join(cache_dir, f"deepgrow_{self.dimension}D_train")
        logger.info(f"Preparing Dataset for Deepgrow-{self.dimension}D:: {output_dir}")

        datalist = create_dataset(
            datalist=datastore.datalist(),
            base_dir=None,
            output_dir=output_dir,
            dimension=self.dimension,
            pixdim=[1.0] * self.dimension,
        )

        logging.info(f"+++ Total Records: {len(datalist)}")
        return datalist

    def cleanup(self, request):
        # run_id = request["run_id"]
        output_dir = os.path.join(pathlib.Path.home(), ".cache", "monailabel", f"deepgrow_{self.dimension}D_train")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

    def get_click_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanced(guidance="guidance", discrepancy="discrepancy", probability="probability"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self, context: Context):
        # Dataset preparation
        t: List[Any] = [
            LoadImaged(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            SpatialCropForegroundd(keys=("image", "label"), source_key="label", spatial_size=self.roi_size),
            Resized(keys=("image", "label"), spatial_size=self.model_size, mode=("area", "nearest")),
            NormalizeIntensityd(keys="image", subtrahend=208.0, divisor=388.0),  # type: ignore
        ]
        if self.dimension == 3:
            t.append(FindAllValidSlicesd(label="label", sids="sids"))
        t.extend(
            [
                AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
                AddGuidanceSignald(image="image", guidance="guidance"),
                EnsureTyped(keys=("image", "label"), device=context.device),
                SelectItemsd(keys=("image", "label", "guidance")),
            ]
        )
        return t

    def train_post_transforms(self, context: Context):
        return [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]

    def val_pre_transforms(self, context: Context):
        return self.train_pre_transforms(context)

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_iteration_update(self, context: Context):
        return Interaction(
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_train_interactions,
            key_probability="probability",
            train=True,
        )

    def val_iteration_update(self, context: Context):
        return Interaction(
            transforms=self.get_click_transforms(context),
            max_interactions=self.max_val_interactions,
            key_probability="probability",
            train=False,
        )
