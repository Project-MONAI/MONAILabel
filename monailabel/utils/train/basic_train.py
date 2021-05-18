import logging
import os
from abc import abstractmethod

import torch
from monai.data import DataLoader, PersistentDataset, partition_dataset
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss

from monailabel.utils.train import TrainTask

logger = logging.getLogger(__name__)


class BasicTrainTask(TrainTask):
    """
    This provides Basic Train Task to train segmentation models over MSD Dataset.
    """

    def __init__(
        self,
        output_dir,
        data_list,
        network,
        load_path=None,
        load_dict=None,
        val_split=0.2,
        device="cuda",
        lr=0.0001,
        train_batch_size=1,
        train_num_workers=1,
        train_save_interval=50,
        val_interval=1,
        val_batch_size=1,
        val_num_workers=1,
        final_filename="checkpoint_final.pt",
        key_metric_filename="model.pt",
    ):
        """

        :param output_dir: Output to save the model checkpoints, events etc...
        :param data_list: List of dictionary that normally contains {image, label}
        :param network: If None then UNet with channels(16, 32, 64, 128, 256) is used
        :param load_path: Initialize model from existing checkpoint
        :param load_dict: Provide dictionary to load from checkpoint.  If None, then `net` will be loaded
        :param val_split: Split ratio for validation dataset if `validation` field is not found in `data_list`
        :param device: device name
        :param lr: Learning Rate (LR)
        :param train_batch_size: train batch size
        :param train_num_workers: number of workers for training
        :param train_save_interval: checkpoint save interval for training
        :param val_interval: validation interval (run every x epochs)
        :param val_batch_size: batch size for validation step
        :param val_num_workers: number of workers for validation step
        :param final_filename: name of final checkpoint that will be saved
        """
        super().__init__()
        self.output_dir = output_dir
        self._train_datalist, self._val_datalist = (
            partition_dataset(data_list, ratios=[(1 - val_split), val_split], shuffle=True)
            if val_split > 0.0
            else (data_list, [])
        )

        logger.info(f"Total Records for Training: {len(self._train_datalist)}/{len(data_list)}")
        logger.info(f"Total Records for Validation: {len(self._val_datalist)}/{len(data_list)}")

        self._device = torch.device(device)
        self._network = network
        self._load_path = load_path
        self._load_dict = load_dict

        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        self._train_batch_size = train_batch_size
        self._train_num_workers = train_num_workers
        self._train_save_interval = train_save_interval

        self._val_interval = val_interval
        self._val_batch_size = val_batch_size
        self._val_num_workers = val_num_workers
        self._final_filename = final_filename
        self._key_metric_filename = key_metric_filename

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
            dataset=PersistentDataset(self._train_datalist, self.train_pre_transforms(), cache_dir=None),
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=self._train_num_workers,
        )

    def train_inferer(self):
        return SimpleInferer()

    def train_key_metric(self):
        return {"train_dice": MeanDice(output_transform=lambda x: (x["pred"], x["label"]))}

    def train_handlers(self):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer(), step_size=5000, gamma=0.1)

        handlers = [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
            TensorBoardStatsHandler(
                log_dir=self.output_dir,
                tag_name="train_loss",
                output_transform=lambda x: x["loss"],
            ),
            CheckpointSaver(
                save_dir=self.output_dir,
                save_dict={"model": self.network(), "optimizer": self.optimizer()},
                save_interval=self._train_save_interval,
                save_final=True,
                final_filename=self._final_filename,
                save_key_metric=True,
                key_metric_filename=self._key_metric_filename,
            ),
        ]

        eval = self.evaluator()
        if eval:
            logger.info(f"Adding Validation Handler to run every '{self._val_interval}' interval")
            handlers.append(ValidationHandler(validator=eval, interval=self._val_interval, epoch_level=True))

        logger.info(f"Load Path {self._load_path}")
        if self._load_path and os.path.exists(self._load_path):
            handlers.append(
                CheckpointLoader(
                    load_path=self._load_path,
                    load_dict={"model": self.network()} if self._load_dict is None else self._load_dict,
                )
            )

        return handlers

    def train_additional_metrics(self):
        return None

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

    def val_pre_transforms(self):
        return self.train_pre_transforms()

    def val_post_transforms(self):
        return self.train_post_transforms()

    def val_handlers(self):
        return [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=self.output_dir, output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=self.output_dir,
                save_dict={"net": self.network()},
                save_key_metric=True,
            ),
        ]

    def val_key_metric(self):
        return {"val_mean_dice": MeanDice(output_transform=lambda x: (x["pred"], x["label"]))}

    def train_iteration_update(self):
        return None

    def val_iteration_update(self):
        return None

    def event_names(self):
        return None

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
