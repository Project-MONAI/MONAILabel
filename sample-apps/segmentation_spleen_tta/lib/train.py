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

import glob
import logging
import os

import torch
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    Resized,
    Spacingd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train Segmentation model for spleen",
        **kwargs,
    ):
        self._network = network
        super().__init__(model_dir, description, **kwargs)

    def network(self):
        return self._network

    def optimizer(self):
        return torch.optim.Adam(self._network.parameters(), lr=0.0001)

    def loss_function(self):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def train_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            Resized(keys=("image", "label"), spatial_size=(128, 128, 128), mode=("area", "nearest")),
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ]

    def val_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            Resized(keys=("image", "label"), spatial_size=(128, 128, 128), mode=("area", "nearest")),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self):
        return SimpleInferer()

    def partition_datalist(self, request, datalist, shuffle=True):

        # Training images
        train_d = datalist

        # Validation images
        data_dir = "/home/adp20local/Documents/Datasets/monailabel_datasets/spleen/validation_imgs"
        val_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        val_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        val_d = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]

        return train_d, val_d
