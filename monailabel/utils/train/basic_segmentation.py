import logging
from abc import abstractmethod

import torch

from monai.data import DataLoader, PersistentDataset, partition_dataset
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointSaver,
    MeanDice,
    ValidationHandler,
    LrScheduleHandler)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monailabel.utils.train import TrainTask

logger = logging.getLogger(__name__)


class BasicTrainSegmentationTask(TrainTask):
    """
    This provides Basic Train Engine to train segmentation models over MSD Dataset.
    """

    def __init__(
            self,
            output_dir,
            data_list,
            network,
            val_split=0.2,
            device='cuda',
            lr=0.0001,
            train_batch_size=4,
            train_num_workers=4,
            train_save_interval=400,
            val_interval=1,
            val_batch_size=1,
            val_num_workers=1,
    ):
        """

        :param output_dir: Output to save the model checkpoints, events etc...
        :param data_list: List of dictionary that normally contains {image, label}
        :param network: If None then UNet with channels(16, 32, 64, 128, 256) is used
        :param val_split: Split ratio for validation dataset if `validation` field is not found in `data_list`
        :param device: device name
        :param lr: Learning Rate (LR)
        :param train_batch_size: train batch size
        :param train_num_workers: number of workers for training
        :param train_save_interval: checkpoint save interval for training
        :param val_interval: validation interval (run every x epochs)
        :param val_batch_size: batch size for validation step
        :param val_num_workers: number of workers for validation step
        """
        super().__init__()
        self._output_dir = output_dir
        self._train_datalist, self._val_datalist = partition_dataset(
            data_list,
            ratios=[(1 - val_split), val_split],
            shuffle=True
        )

        logger.info(f"Total Records for Training: {len(self._train_datalist)}")
        logger.info(f"Total Records for Validation: {len(self._val_datalist)}")

        self._device = torch.device(device)
        self._network = network

        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        self._train_batch_size = train_batch_size
        self._train_num_workers = train_num_workers
        self._train_save_interval = train_save_interval

        self._val_interval = val_interval
        self._val_batch_size = val_batch_size
        self._val_num_workers = val_num_workers

    def device(self):
        return self._device

    def network(self):
        return self._network

    def loss_function(self):
        return self._loss_function

    def optimizer(self):
        return self._optimizer

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

    def val_data_loader(self):
        return DataLoader(
            dataset=PersistentDataset(self._val_datalist, self.val_pre_transforms()),
            batch_size=self._val_batch_size,
            shuffle=False,
            num_workers=self._val_num_workers)

    def val_pre_transforms(self):
        return self.train_pre_transforms()

    def val_post_transforms(self):
        return self.train_post_transforms()

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

    @abstractmethod
    def train_pre_transforms(self):
        pass

    @abstractmethod
    def train_post_transforms(self):
        pass

    @abstractmethod
    def val_inferer(self):
        pass
