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

import torch
from monai.apps.deepedit.transforms import NormalizeLabelsInDatasetd
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class SegmentationSam(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train SAM model",
        **kwargs,
    ):
        self._network = network
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(self.network.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    def loss_function(self, context: Context):
        return DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def train_pre_transforms(self, context: Context):
        # TODO: need data loader
        return [
        ]

    def train_post_transforms(self, context: Context):
        return [
        ]

    def val_pre_transforms(self, context: Context):
        return [
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer
