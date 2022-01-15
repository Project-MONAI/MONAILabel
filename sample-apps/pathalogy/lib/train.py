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
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    LoadImaged,
    EnsureChannelFirstd, AddChanneld, ScaleIntensityd, RandRotate90d, EnsureTyped, RandCropByPosNegLabeld,
)
from torchvision import transforms  # noqa

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

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
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys="image"),
            AddChanneld(keys="label"),

            # ToTensorD(keys="image"),
            # TorchVisionD(
            #     keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            # ),
            # ToNumpyD(keys="image"),

            # RandFlipD(keys="image", prob=0.5),
            # RandRotate90D(keys="image", prob=0.5),
            # CastToTypeD(keys="image", dtype=np.float32),
            # RandZoomD(keys="image", prob=0.5, min_zoom=0.9, max_zoom=1.1),
            # ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),

            ScaleIntensityd(keys=("image", "label")),
            RandCropByPosNegLabeld(
                keys=("image", "label"), label_key="label", spatial_size=(512, 512), pos=1, neg=1, num_samples=4
            ),

            RandRotate90d(keys=("image", "label"), prob=0.5, spatial_axes=[0, 1]),
            EnsureTyped(keys=("image", "label")),
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

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys="image"),
            AddChanneld(keys="label"),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=(512, 512), sw_batch_size=4, overlap=0.25)
