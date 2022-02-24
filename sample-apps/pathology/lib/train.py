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
import os

import numpy as np
import torch
from ignite.metrics import Accuracy
from monai.handlers import from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    RandRotate90d,
    ScaleIntensityRangeD,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
)

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from .flatten_pannuke import split_pannuke_dataset
from .handlers import TensorBoardImageHandler
from .transforms import EncodeLabelChannelsd, FilterImaged

logger = logging.getLogger(__name__)


class TrainSegmentation(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        label_channels,
        description="Pathology Semantic Segmentation for Nuclei (PanNuke Dataset)",
        **kwargs,
    ):
        self._network = network
        self.labels = labels
        self.label_channels = label_channels
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(self._network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)

    def pre_process(self, request, datastore: Datastore):
        self.cleanup(request)
        cache_dir = os.path.join(self.get_cache_dir(request), "train_ds")

        ds = datastore.datalist()
        if len(ds) == 1:
            image = np.load(ds[0]["image"])
            if len(image.shape) > 3:
                ds = split_pannuke_dataset(ds[0]["image"], ds[0]["label"], cache_dir)

        logging.info("+++ Total Records: {}".format(len(ds)))
        return ds

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            FilterImaged(keys="image", min_size=0),
            AsChannelFirstd(keys="image"),
            EncodeLabelChannelsd(keys="label", labels=self.labels, label_channels=self.label_channels),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            RandRotate90d(keys=("image", "label"), prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            EnsureTyped(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=(len(self.labels) + 1, len(self.labels) + 1),
            ),
        ]

    def train_key_metric(self, context: Context):
        return {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_key_metric(self, context: Context):
        return {"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir))
        return handlers