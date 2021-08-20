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

import copy
import json
import logging
import os
import time
from abc import abstractmethod
from datetime import datetime
from typing import List

import torch
from monai.data import CacheDataset, DataLoader, PersistentDataset, partition_dataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
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
from monai.transforms import Compose

from monailabel.interfaces import Datastore
from monailabel.interfaces.tasks import TrainTask
from monailabel.utils.train.handler import PublishStatsAndModel, prepare_stats

logger = logging.getLogger(__name__)


class BasicTrainTask(TrainTask):
    """
    This provides Basic Train Task to train a model using SupervisedTrainer and SupervisedEvaluator from MONAI
    """

    def __init__(
        self,
        model_dir,
        description=None,
        config=None,
        amp=True,
        load_path=None,
        load_dict=None,
        publish_path=None,
        stats_path=None,
        train_save_interval=50,
        val_interval=1,
        final_filename="checkpoint_final.pt",
        key_metric_filename="model.pt",
    ):
        """
        :param model_dir: Base Model Dir to save the model checkpoints, events etc...
        :param description: Description for this task
        :param config: K,V pairs to be part of user config
        :param amp: Enable AMP for training
        :param load_path: Initialize model from existing checkpoint (pre-trained)
        :param load_dict: Provide dictionary to load from checkpoint.  If None, then `net` will be loaded
        :param publish_path: Publish path for best trained model (based on best key metric)
        :param stats_path: Path to save the train stats
        :param train_save_interval: checkpoint save interval for training
        :param val_interval: validation interval (run every x epochs)
        :param final_filename: name of final checkpoint that will be saved
        :param key_metric_filename: best key metric model file name
        """
        super().__init__(description)

        self._model_dir = model_dir
        self._amp = amp
        self._config = {
            "name": "model_01",
            "pretrained": True,
            "device": "cuda",
            "max_epochs": 50,
            "val_split": 0.2,
            "train_batch_size": 1,
            "val_batch_size": 1,
        }
        if config:
            self._config.update(config)

        self._load_path = load_path
        self._load_dict = load_dict
        self._publish_path = publish_path
        self._stats_path = stats_path if stats_path else os.path.join(model_dir, "train_stats.json")

        self._train_save_interval = train_save_interval
        self._val_interval = val_interval
        self._final_filename = final_filename
        self._key_metric_filename = key_metric_filename

    @abstractmethod
    def network(self):
        pass

    @abstractmethod
    def optimizer(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass

    def train_data_loader(self, datalist, batch_size=1, num_workers=0, cached=False):
        transforms = self._validate_transforms(self.train_pre_transforms(), "Training", "pre")
        dataset = CacheDataset(datalist, transforms) if cached else PersistentDataset(datalist, transforms, None)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def train_inferer(self):
        return SimpleInferer()

    def train_key_metric(self):
        return {"train_dice": MeanDice(output_transform=from_engine(["pred", "label"]))}

    def load_path(self, output_dir, pretrained=True):
        load_path = os.path.join(output_dir, self._key_metric_filename)
        if not os.path.exists(load_path) and pretrained:
            load_path = self._load_path
        return load_path

    def train_handlers(self, output_dir, events_dir, evaluator):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer(), step_size=5000, gamma=0.1)

        handlers = [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
            TensorBoardStatsHandler(
                log_dir=events_dir,
                tag_name="train_loss",
                output_transform=from_engine(["loss"], first=True),
            ),
            CheckpointSaver(
                save_dir=output_dir,
                save_dict={"model": self.network()},
                save_interval=self._train_save_interval,
                save_final=True,
                final_filename=self._final_filename,
                save_key_metric=True,
                key_metric_filename=self._key_metric_filename,
            ),
        ]

        if evaluator:
            logger.info(f"Adding Validation Handler to run every '{self._val_interval}' interval")
            handlers.append(ValidationHandler(validator=evaluator, interval=self._val_interval, epoch_level=True))

        return handlers

    def train_additional_metrics(self):
        return None

    def val_data_loader(self, datalist, batch_size=1, num_workers=0, cached=False):
        transforms = self._validate_transforms(self.val_pre_transforms(), "Validation", "pre")
        dataset = CacheDataset(datalist, transforms) if cached else PersistentDataset(datalist, transforms, None)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def val_pre_transforms(self):
        return self.train_pre_transforms()

    def val_post_transforms(self):
        return self.train_post_transforms()

    def val_handlers(self, output_dir, events_dir):
        return [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=events_dir, output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=output_dir,
                save_dict={"model": self.network()},
                save_key_metric=True,
                key_metric_filename=f"eval_{self._key_metric_filename}",
            ),
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

    def partition_datalist(self, request, datalist, shuffle=True):
        val_split = request["val_split"]
        if val_split > 0.0:
            return partition_dataset(datalist, ratios=[(1 - val_split), val_split], shuffle=shuffle)
        return datalist, []

    def stats(self):
        if self._stats_path and os.path.exists(self._stats_path):
            with open(self._stats_path, "r") as fc:
                return json.load(fc)
        return {}

    def config(self):
        return self._config

    @staticmethod
    def _validate_transforms(transforms, step="Training", name="pre"):
        if not transforms or isinstance(transforms, Compose):
            return transforms
        if isinstance(transforms, list):
            return Compose(transforms)
        raise ValueError(f"{step} {name}-transforms are not of `list` or `Compose` type")

    def __call__(self, request, datastore: Datastore):
        start_ts = time.time()
        req = copy.deepcopy(self._config)
        req.update(copy.deepcopy(request))
        logger.info(f"Train Request (input): {request}")
        logger.info(f"Train Request (final): {req}")

        name = req["name"]
        device = torch.device(req["device"] if torch.cuda.is_available() else "cpu")
        max_epochs = req["max_epochs"]
        train_batch_size = req["train_batch_size"]
        val_batch_size = req["val_batch_size"]
        pretrained = req["pretrained"]

        output_dir = os.path.join(self._model_dir, name)
        run_id = datetime.now().strftime("%Y%m%d_%H%M")
        events_dir = os.path.join(output_dir, f"events_{run_id}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        train_ds, val_ds = self.partition_datalist(req, datastore.datalist())
        logger.info(f"Total Records for Training: {len(train_ds)}")
        logger.info(f"Total Records for Validation: {len(val_ds)}")

        publisher = PublishStatsAndModel(
            self._stats_path, self._publish_path, self._key_metric_filename, start_ts, run_id, output_dir, None, None
        )
        evaluator = None
        if val_ds and len(val_ds) > 0:
            val_hanlders: List = self.val_handlers(output_dir, events_dir)
            val_hanlders.append(publisher)

            evaluator = SupervisedEvaluator(
                device=device,
                val_data_loader=self.val_data_loader(val_ds, val_batch_size),
                network=self.network().to(device),
                inferer=self.val_inferer(),
                postprocessing=self._validate_transforms(self.val_post_transforms(), "Validation", "post"),
                key_val_metric=self.val_key_metric(),
                additional_metrics=self.val_additional_metrics(),
                val_handlers=val_hanlders,
                iteration_update=self.val_iteration_update(),
                event_names=self.event_names(),
            )

        train_handlers: List = self.train_handlers(output_dir, events_dir, evaluator)
        if not evaluator:
            train_handlers.append(publisher)

        load_path = self.load_path(output_dir, pretrained)
        if load_path and os.path.exists(load_path):
            logger.info(f"Load Path {load_path}")
            train_handlers.append(
                CheckpointLoader(
                    load_path=load_path,
                    load_dict={"model": self.network()} if self._load_dict is None else self._load_dict,
                )
            )

        trainer = SupervisedTrainer(
            device=device,
            max_epochs=max_epochs,
            train_data_loader=self.train_data_loader(train_ds, train_batch_size),
            network=self.network().to(device),
            optimizer=self.optimizer(),
            loss_function=self.loss_function(),
            inferer=self.train_inferer(),
            amp=self._amp,
            postprocessing=self._validate_transforms(self.train_post_transforms(), "Training", "post"),
            key_train_metric=self.train_key_metric(),
            train_handlers=train_handlers,
            iteration_update=self.train_iteration_update(),
            event_names=self.event_names(),
        )

        publisher.trainer = trainer
        publisher.evaluator = evaluator

        # Run Training
        trainer.run()

        # Try to clear cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return prepare_stats(start_ts, trainer, evaluator)
