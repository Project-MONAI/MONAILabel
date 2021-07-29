import logging

import numpy as np
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandShiftIntensityd,
    Resized,
    Spacingd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
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
            CropForegroundd(keys=("image", "label"), source_key="image"),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=(256, 256, 128),
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1),
            ),
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self):
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=True,
                n_classes=2,
            ),
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
            CropForegroundd(keys=("image", "label"), source_key="image"),
            Resized(keys=("image", "label"), spatial_size=(256, 256, 128), mode=("area", "nearest")),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self):
        return SimpleInferer()
