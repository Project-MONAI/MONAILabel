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

from ignite.metrics import Accuracy
from monai.handlers import from_engine
from monai.inferers import SimpleInferer
from monai.optimizers import Novograd
from monai.transforms import Activationsd, AsDiscreted
from torch.nn import BCEWithLogitsLoss

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from .handlers import TensorBoardImageHandler
from .transforms import ImageToGridd

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        image_size=1024,
        patch_size=64,
        description="Pathology Segmentation model",
        **kwargs,
    ):
        self._network = network
        self._image_size = image_size
        self._patch_size = patch_size
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return Novograd(self._network.parameters(), 0.001)

    def loss_function(self, context: Context):
        return BCEWithLogitsLoss()

    def train_key_metric(self, context: Context):
        return {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def train_pre_transforms(self, context: Context):
        return [
            ImageToGridd(keys=("image", "label"), image_size=self._image_size, patch_size=self._patch_size),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]

    def val_key_metric(self, context: Context):
        return {"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir))
        return handlers
