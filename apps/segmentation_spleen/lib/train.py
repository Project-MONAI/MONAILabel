import logging

import torch
from monai.data import load_decathlon_datalist, DataLoader, PersistentDataset
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointSaver,
    MeanDice,
    ValidationHandler,
    LrScheduleHandler)
from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.losses import DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet, BasicUNet
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    ScaleIntensityRanged,
    Activationsd,
    AsDiscreted,
    Compose, CropForegroundd, RandCropByPosNegLabeld, RandShiftIntensityd, ToTensord
)

from server.interface import TrainEngine

logger = logging.getLogger(__name__)


class SpleenTrainEngine(TrainEngine):
    def __init__(self, request):
        super().__init__()
        self._output_dir = request['output_dir']

        data_list = request['data_list']
        data_root = request.get('data_root')
        self._train_datalist = load_decathlon_datalist(data_list, True, "training", data_root)
        self._val_datalist = load_decathlon_datalist(data_list, True, "validation", data_root)

        logger.info(f"Total Records for Training: {len(self._train_datalist)}")
        logger.info(f"Total Records for Validation: {len(self._val_datalist)}")

        self._device = torch.device(request.get('device', 'cuda'))

        if request.get("network") == "BasicUNET":
            self._network = BasicUNet(dimensions=3, in_channels=1, out_channels=2, features=(16, 32, 64, 128, 256, 16))
        else:
            self._network = UNet(
                dimensions=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH
            )

        self._optimizer = torch.optim.Adam(self._network.parameters(), request.get('lr', 0.0001))
        self._loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        train = request.get('train', {})
        self._train_batch_size = train.get('batch_size', 4)
        self._train_num_workers = train.get('num_workers', 4)
        self._train_save_interval = train.get('save_interval', 400)

        val = request.get('val', {})
        self._val_batch_size = val.get('batch_size', 1)
        self._val_num_workers = val.get('num_workers', 1)
        self._val_interval = val.get('interval', 1)

    def device(self):
        return self._device

    def network(self):
        return self._network

    def loss_function(self):
        return self._loss_function

    def optimizer(self):
        return self._optimizer

    def train_pre_transforms(self):
        return Compose([
            LoadImaged(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            RandCropByPosNegLabeld(keys=("image", "label"), label_key="label", spatial_size=(96, 96, 96), pos=1,
                                   neg=1, num_samples=4, image_key="image", image_threshold=0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=("image", "label"))
        ])

    def train_post_transforms(self):
        return Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)
        ])

    def train_data_loader(self):
        return DataLoader(
            dataset=PersistentDataset(self._train_datalist, self.train_pre_transforms()),
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=self._train_num_workers)

    def train_inferer(self):
        return SimpleInferer()

    def train_key_metric(self):
        return {"train_dice": MeanDice(output_transform=lambda x: (x["pred"], x["label"]))}

    def train_handlers(self):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer(), step_size=5000, gamma=0.1)

        return [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            ValidationHandler(validator=self.evaluator(), interval=self._val_interval, epoch_level=True),
            StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
            TensorBoardStatsHandler(log_dir=self._output_dir, tag_name="train_loss",
                                    output_transform=lambda x: x["loss"]),
            CheckpointSaver(save_dir=self._output_dir, save_dict={"net": self.network(), "opt": self.optimizer()},
                            save_interval=self._train_save_interval),
        ]

    def train_additional_metrics(self):
        return None

    def val_pre_transforms(self):
        return Compose([
            LoadImaged(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            ToTensord(keys=("image", "label"))
        ])

    def val_post_transforms(self):
        return self.train_post_transforms()

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(160, 160, 160), sw_batch_size=1, overlap=0.25)

    def val_data_loader(self):
        return DataLoader(
            dataset=PersistentDataset(self._val_datalist, self.val_pre_transforms()),
            batch_size=self._val_batch_size,
            shuffle=False,
            num_workers=self._val_num_workers)

    def val_handlers(self):
        return [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=self._output_dir, output_transform=lambda x: None),
            CheckpointSaver(save_dir=self._output_dir, save_dict={"net": self.network()}, save_key_metric=True)
        ]

    def val_key_metric(self):
        return {"val_mean_dice": MeanDice(output_transform=lambda x: (x["pred"], x["label"]))}

    def val_additional_metrics(self):
        return None
