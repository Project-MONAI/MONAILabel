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
                pixdim=(1.5, 1.5, 2.0),
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
            Spacingd(
                keys=("image", "label"),
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(160, 160, 160), sw_batch_size=1, overlap=0.5)

    #TODO Define the data loaders below again with Caching if you want this code to go faster
    '''
    def train_data_loader(self):

        if isinstance(self.train_pre_transforms(), list):
            train_pre_transforms = Compose(self.train_pre_transforms())
        elif isinstance(self.train_pre_transforms(), Compose):
            train_pre_transforms = self.train_pre_transforms()
        else:
            raise ValueError("Training pre-transforms are not of `list` or `Compose` type")

        return DataLoader(
            dataset=PersistentDataset(self._train_datalist, train_pre_transforms, cache_dir=None),
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=self._train_num_workers,
        )

    def val_data_loader(self):
        return (
            DataLoader(
                dataset=PersistentDataset(self._val_datalist, self.val_pre_transforms(), cache_dir=None),
                batch_size=self._val_batch_size,
                shuffle=False,
                num_workers=self._val_num_workers,
            )
            if self._val_datalist and len(self._val_datalist) > 0
            else None
        )
    '''