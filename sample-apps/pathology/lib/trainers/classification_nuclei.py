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

import numpy as np
import torch
from ignite.metrics import Accuracy
from lib.handlers import TensorBoardImageHandler
from lib.transforms import FilterImaged, FilterLabelByClassd, FromClassd
from lib.utils import split_dataset, split_nuclei_dataset
from monai.apps.nuclick.transforms import ExtractPatchd, FlattenLabeld
from monai.handlers import from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    RandRotate90d,
    ScaleIntensityRangeD,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
)
from tqdm import tqdm

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class ClassificationNuclei(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        tile_size=(256, 256),
        patch_size=128,
        min_area=5,
        description="Pathology Classification Nuclei",
        **kwargs,
    ):
        self._network = network
        self.labels = labels
        self.tile_size = tile_size
        self.patch_size = patch_size
        self.min_area = min_area
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)

    def pre_process(self, request, datastore: Datastore):
        self.cleanup(request)

        cache_dir = os.path.join(self.get_cache_dir(request), "train_ds")
        source = request.get("dataset_source")
        max_region = request.get("dataset_max_region", (10240, 10240))
        max_region = (max_region, max_region) if isinstance(max_region, int) else max_region[:2]

        ds = split_dataset(
            datastore=datastore,
            cache_dir=cache_dir,
            source=source,
            groups=self.labels,
            tile_size=self.tile_size,
            max_region=max_region,
            limit=request.get("dataset_limit", 0),
            randomize=request.get("dataset_randomize", True),
        )
        logger.info(f"Split data (len: {len(ds)}) based on each nuclei")
        ds_new = []
        limit = request.get("dataset_limit", 0)
        for d in tqdm(ds):
            ds_new.extend(split_nuclei_dataset(d, min_area=self.min_area))
            if 0 < limit < len(ds_new):
                ds_new = ds_new[:limit]
                break
        return ds_new

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            FilterImaged(keys="image", min_size=80),
            FlattenLabeld(keys="label"),
            AsChannelFirstd(keys="image"),
            AddChanneld(keys="label"),
            FilterLabelByClassd(keys="label", key_class="class"),
            ExtractPatchd(keys=("image", "label"), patch_size=self.patch_size),
            CropForegroundd(keys=("image", "label"), source_key="label"),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            FromClassd(keys="label", key_class="class"),
            ToTensord(keys="label"),
            RandRotate90d(keys="image", prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            EnsureTyped(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", softmax=len(self.labels) > 1, sigmoid=len(self.labels) == 1),
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
