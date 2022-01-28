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

from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.optimizers import Novograd
from monai.transforms import Activationsd, AsDiscreted, EnsureTyped

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from .transforms import ImageToGridd

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Pathology Segmentation model",
        **kwargs,
    ):
        self._network = network
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return Novograd(self._network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, batch=True)

    def train_pre_transforms(self, context: Context):
        return [
            ImageToGridd(keys=("image", "label"), image_size=4096, patch_size=256),
        ]

    def train_post_transforms(self, context: Context):
        return [
            EnsureTyped(keys=("pred", "label")),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=True,
                n_classes=2,
            ),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=(256, 256), sw_batch_size=16, overlap=0.25)
