import filecmp
import json
import logging
import os
import shutil
from abc import abstractmethod
from datetime import datetime

import torch
from monai.data import DataLoader, PersistentDataset
from monai.engines.workflow import Engine, Events
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms.compose import Compose

from monailabel.interfaces.tasks import TrainTask

logger = logging.getLogger(__name__)


class BasicTrainTask(TrainTask):
    """
    This provides Basic Train Task to train segmentation models over MSD Dataset.
    """

    def __init__(
        self,
        output_dir,
        train_datalist,
        val_datalist,
        network,
        load_path=None,
        load_dict=None,
        publish_path=None,
        stats_path=None,
        device="cuda",
        max_epochs=1,
        amp=True,
        lr=0.0001,
        train_batch_size=1,
        train_num_workers=0,
        train_save_interval=50,
        val_interval=1,
        val_batch_size=1,
        val_num_workers=0,
        final_filename="checkpoint_final.pt",
        key_metric_filename="model.pt",
        **kwargs,
    ):
        """

        :param output_dir: Output to save the model checkpoints, events etc...
        :param train_datalist: Training List of dictionary that normally contains {image, label}
        :param val_datalist: Validation List of dictionary that normally contains {image, label}
        :param network: If None then UNet with channels(16, 32, 64, 128, 256) is used
        :param load_path: Initialize model from existing checkpoint
        :param load_dict: Provide dictionary to load from checkpoint.  If None, then `net` will be loaded
        :param publish_path: Publish path for best trained model (based on best key metric)
        :param stats_path: Path to save the train stats
        :param device: device name
        :param max_epochs: maximum epochs to run
        :param amp: use amp
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
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M")
        self.events_dir = os.path.join(output_dir, f"events_{self.run_id}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self._train_datalist = train_datalist
        self._val_datalist = val_datalist if val_datalist else []

        logger.info(f"Total Records for Training: {len(self._train_datalist)}")
        logger.info(f"Total Records for Validation: {len(self._val_datalist)}")

        self._device = torch.device(device)
        self._max_epochs = max_epochs
        self._amp = amp
        self._network = network
        self._load_path = load_path
        self._load_dict = load_dict
        self.publish_path = publish_path
        self.stats_path = stats_path

        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        self._train_batch_size = train_batch_size
        self._train_num_workers = train_num_workers
        self._train_save_interval = train_save_interval

        self._val_interval = val_interval
        self._val_batch_size = val_batch_size
        self._val_num_workers = val_num_workers
        self._final_filename = final_filename
        self.key_metric_filename = key_metric_filename

    def device(self):
        return self._device

    def max_epochs(self):
        return self._max_epochs

    def amp(self):
        return self._amp

    def network(self):
        return self._network

    def loss_function(self):
        return self._loss_function

    def optimizer(self):
        return self._optimizer

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

    def train_inferer(self):
        return SimpleInferer()

    def train_key_metric(self):
        return {"train_dice": MeanDice(output_transform=from_engine(["pred", "label"]))}

    def train_handlers(self):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer(), step_size=5000, gamma=0.1)

        handlers = [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
            TensorBoardStatsHandler(
                log_dir=self.events_dir,
                tag_name="train_loss",
                output_transform=from_engine(["loss"], first=True),
            ),
            CheckpointSaver(
                save_dir=self.output_dir,
                save_dict={"model": self.network()},
                save_interval=self._train_save_interval,
                save_final=True,
                final_filename=self._final_filename,
                save_key_metric=True,
                key_metric_filename=self.key_metric_filename,
            ),
        ]

        e = self.evaluator()
        if e:
            logger.info(f"Adding Validation Handler to run every '{self._val_interval}' interval")
            handlers.append(ValidationHandler(validator=e, interval=self._val_interval, epoch_level=True))
        else:
            handlers.append(PublishStatsAndModel(parent=self))

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

        if isinstance(self.train_pre_transforms(), list):
            val_pre_transforms = Compose(self.train_pre_transforms())
        elif isinstance(self.train_pre_transforms(), Compose):
            val_pre_transforms = self.train_pre_transforms()
        else:
            raise ValueError("Validation pre-transforms are not of `list` or `Compose` type")

        return val_pre_transforms

    def val_post_transforms(self):

        if isinstance(self.train_post_transforms(), list):
            val_post_transforms = Compose(self.train_post_transforms())
        elif isinstance(self.train_post_transforms(), Compose):
            val_post_transforms = self.train_post_transforms()
        else:
            raise ValueError("Validation pre-transforms are not of `list` or `Compose` type")

        return val_post_transforms

    def val_handlers(self):
        return [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=self.events_dir, output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=self.output_dir,
                save_dict={"model": self.network()},
                save_key_metric=True,
                key_metric_filename=f"eval_{self.key_metric_filename}",
            ),
            PublishStatsAndModel(parent=self),
        ]

    def val_key_metric(self):
        return {"val_mean_dice": MeanDice(output_transform=from_engine(["pred", "label"]))}

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


class PublishStatsAndModel:
    def __init__(self, parent: BasicTrainTask):
        self.parent: BasicTrainTask = parent

    def iteration_completed(self):
        filename = datetime.now().strftime(f"stats_{self.parent.run_id}.json")
        filename = os.path.join(self.parent.output_dir, filename)

        stats = self.parent.prepare_stats()
        with open(filename, "w") as f:
            json.dump(stats, f, indent=2)

        if self.parent.stats_path:
            shutil.copy(filename, self.parent.stats_path)

        publish_path = self.parent.publish_path
        if publish_path:
            final_model = os.path.join(self.parent.output_dir, self.parent.key_metric_filename)
            if os.path.exists(final_model):
                if not os.path.exists(publish_path) or not filecmp.cmp(publish_path, final_model):
                    shutil.copy(final_model, publish_path)
                    logger.info(f"New Model published: {final_model} => {publish_path}")
        return stats

    def attach(self, engine: Engine) -> None:
        if not engine.has_event_handler(self.iteration_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.iteration_completed)
