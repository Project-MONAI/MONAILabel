import logging

import numpy as np
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandHistogramShiftd,
    RandShiftIntensityd,
    Resized,
    Spacingd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def train_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image"),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandAdjustContrastd(keys="image", gamma=6),
                RandHistogramShiftd(keys="image", num_control_points=8, prob=0.5),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=1.0,
                    spatial_size=(128, 128, 128),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1),
                ),
                Resized(keys=("image", "label"), spatial_size=[128, 128, 128], mode=("area", "nearest")),
                ToTensord(keys=("image", "label")),
            ]
        )

    def train_post_transforms(self):
        return Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=True,
                    n_classes=4,
                ),
            ]
        )

    def val_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image"),
                CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 16]),
                Resized(keys=("image", "label"), spatial_size=[128, 128, 128], mode=("area", "nearest")),
                ToTensord(keys=("image", "label")),
            ]
        )

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.25)
