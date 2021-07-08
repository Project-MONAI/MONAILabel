import logging

from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def train_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self):
        return [
            ToTensord(keys=("pred", "label")),
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
            AddChanneld(keys=("image", "label")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(160, 160, 160), sw_batch_size=1, overlap=0.25)
