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
from typing import Callable, Union

import numpy as np
import torch
from monai.handlers import MeanDice, from_engine
from monai.handlers.ignite_metric import IgniteMetric
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.metrics import MeanIoU
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    DivisiblePadd,
    LoadImaged,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from monai.utils import MetricReduction

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.transform.pre import NormalizeLabeld

logger = logging.getLogger(__name__)


class ToolTracking(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        description="Endoscopy Semantic Segmentation for Tool Tracking",
        **kwargs,
    ):
        self._network = network
        self.labels = labels
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(include_background=False, softmax=True, to_onehot_y=True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            AsChannelFirstd("image"),
            AddChanneld(keys="label"),
            NormalizeLabeld(keys="label"),
            Resized(keys=("image", "label"), spatial_size=(736, 480), mode=("area", "nearest")),
            DivisiblePadd(keys=("image", "label"), k=32),
            ScaleIntensityd(keys=("image", "label")),
            RandRotated(keys=("image", "label"), range_x=np.pi, prob=0.5, mode=["bilinear", "nearest"]),
            RandZoomd(keys=("image", "label"), min_zoom=0.8, max_zoom=1.2, prob=0.5, mode=["bilinear", "nearest"]),
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            ToTensord(keys=("pred", "label")),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=(2, 2),
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            AsChannelFirstd("image"),
            AddChanneld(keys="label"),
            NormalizeLabeld(keys="label"),
            Resized(keys=("image", "label"), spatial_size=(736, 480), mode=("area", "nearest")),
            DivisiblePadd(keys=("image", "label"), k=32),
            ScaleIntensityd(keys=("image", "label")),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def val_key_metric(self, context):
        return {
            self.VAL_KEY_METRIC: MeanDice(include_background=False, output_transform=from_engine(["pred", "label"])),
            "val_mean_iou": MeanIoUMetric(include_background=False, output_transform=from_engine(["pred", "label"])),
        }


class MeanIoUMetric(IgniteMetric):
    def __init__(
        self,
        include_background: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        metric_fn = MeanIoU(include_background=include_background, reduction=reduction)
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)
