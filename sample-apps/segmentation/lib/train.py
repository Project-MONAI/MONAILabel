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
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    ToDeviced,
    ToTensord,
)

from monailabel.deepedit.multilabel.transforms import NormalizeLabelsInDatasetd, SplitPredsLabeld
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        spatial_size=(128, 128, 128),
        label_names=None,
        description="Train generic Segmentation model",
        **kwargs,
    ):
        super().__init__(model_dir, description, **kwargs)
        self._network = network
        self.label_names = label_names
        self.spatial_size = spatial_size

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(self._network.parameters(), lr=0.0001)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, batch=True)

    def train_pre_transforms(self, context: Context):
        t = [
            LoadImaged(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=self.label_names),
            AddChanneld(keys=("image", "label")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=("image", "label"),
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys="image",
                offsets=0.10,
                prob=0.50,
            ),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            EnsureTyped(keys=("image", "label")),
        ]
        return t

    def train_post_transforms(self, context: Context):
        return [
            ToTensord(keys=("pred", "label")),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=(True, True),
                n_classes=len(self.label_names),
            ),
            SplitPredsLabeld(keys="pred"),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=self.label_names),
            AddChanneld(keys=("image", "label")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys=("image"),
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            EnsureTyped(keys=("image", "label")),
            ToDeviced(keys=("image", "label"), device=context.device),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def partition_datalist(self, context: Context, shuffle=False):
        # Training images
        train_d = context.datalist

        # Validation images
        data_dir = "/home/adp20local/Documents/Datasets/monailabel_datasets/Slicer/spleen/validation_imgs"
        val_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        val_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        val_d = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]

        if context.local_rank == 0:
            logger.info(f"Total Records for Training: {len(train_d)}")
            logger.info(f"Total Records for Validation: {len(val_d)}")

        return train_d, val_d
