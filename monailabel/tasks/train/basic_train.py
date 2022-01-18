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
import platform
import tempfile
import time
from abc import abstractmethod
from datetime import datetime
from typing import List

import torch
import torch.distributed
from ignite.engine import Events
from ignite.handlers import EarlyStopping
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    PersistentDataset,
    SmartCacheDataset,
    ThreadDataLoader,
    partition_dataset,
)
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
    stopping_fn_from_metric,
)
from monai.inferers import SimpleInferer
from monai.transforms import Compose

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.train.handler import PublishStatsAndModel, prepare_stats
from monailabel.utils.others.generic import remove_file

logger = logging.getLogger(__name__)


class Context:
    def __init__(self):
        self.start_ts = 0  # timestamp
        self.run_id = None  # unique run_id
        self.output_dir = None  # output dir for storing model
        self.events_dir = None  # events dir for storing tensorboard events
        self.datalist = None  # input datalist
        self.train_datalist = None  # train datalist
        self.train_batch_size = None  # train batch size
        self.val_datalist = None  # validation datalist
        self.val_batch_size = None  # validation batch size
        self.device = None  # device on which training will run
        self.network = None  # network
        self.dataset_type = "CacheDataset"  # dataset type
        self.dataloader_type = "ThreadDataLoader"  # dataloader type
        self.pretrained = False  # using pretrained model
        self.max_epochs = 1  # max epochs to run training
        self.multi_gpu = False  # multi gpu enabled
        self.local_rank = 0  # local rank in case of multi gpu
        self.world_size = 0  # world size in case of multi gpu

        self.request = None
        self.trainer = None
        self.evaluator = None


class BasicTrainTask(TrainTask):
    """
    This provides Basic Train Task to train a model using SupervisedTrainer and SupervisedEvaluator from MONAI
    """

    TRAIN_KEY_METRIC = "train_dice"
    VAL_KEY_METRIC = "val_mean_dice"

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
        train_save_interval=20,
        val_interval=1,
        final_filename="checkpoint_final.pt",
        key_metric_filename="model.pt",
        model_dict_key="model",
        find_unused_parameters=False,
        load_strict=False,
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
        :param model_dict_key: key to save network weights into checkpoint
        :param find_unused_parameters: Applicable for DDP/Multi GPU training
        :param load_strict: Load pretrained model in strict mode
        """
        super().__init__(description)

        self._model_dir = model_dir
        self._amp = amp
        self._config = {
            "name": "model_01",
            "pretrained": True,
            "device": "cuda",
            "max_epochs": 50,
            "early_stop_patience": -1,
            "val_split": 0.2,
            "train_batch_size": 1,
            "val_batch_size": 1,
            "multi_gpu": True,
            "gpus": "all",
            "dataset": ["CacheDataset", "PersistentDataset", "SmartCacheDataset", "Dataset"],
            "dataloader": ["ThreadDataLoader", "DataLoader"],
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
        self._model_dict_key = model_dict_key
        self._find_unused_parameters = find_unused_parameters
        self._load_strict = load_strict

    @abstractmethod
    def network(self, context: Context):
        pass

    @abstractmethod
    def optimizer(self, context: Context):
        pass

    @abstractmethod
    def loss_function(self, context: Context):
        pass

    def _dataset(self, context, datalist, replace_rate=0.25):
        if context.multi_gpu:
            world_size = torch.distributed.get_world_size()
            if len(datalist) // world_size:  # every gpu gets full data when datalist is smaller
                datalist = partition_dataset(data=datalist, num_partitions=world_size, even_divisible=True)[
                    context.local_rank
                ]

        transforms = self._validate_transforms(self.train_pre_transforms(context), "Training", "pre")
        dataset = (
            CacheDataset(datalist, transforms)
            if context.dataset_type == "CacheDataset"
            else SmartCacheDataset(datalist, transforms, replace_rate)
            if context.dataset_type == "SmartCacheDataset"
            else PersistentDataset(datalist, transforms, None)
            if context.dataset_type == "PersistentDataset"
            else Dataset(datalist, transforms)
        )
        return dataset, datalist

    def _dataloader(self, context, dataset, batch_size, num_workers):
        if context.dataloader_type == "ThreadDataLoader":
            return ThreadDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def train_data_loader(self, context, num_workers=0):
        dataset, datalist = self._dataset(context, context.train_datalist)
        logger.info(f"{context.local_rank} - Records for Training: {len(datalist)}")
        logger.debug(f"{context.local_rank} - Training: {datalist}")

        return self._dataloader(context, dataset, context.train_batch_size, num_workers)

    def train_inferer(self, context: Context):
        return SimpleInferer()

    def train_key_metric(self, context: Context):
        return {self.TRAIN_KEY_METRIC: MeanDice(output_transform=from_engine(["pred", "label"]))}

    def load_path(self, output_dir, pretrained=True):
        load_path = os.path.join(output_dir, self._key_metric_filename)
        if not os.path.exists(load_path) and pretrained:
            load_path = self._load_path
        return load_path

    def train_handlers(self, context: Context):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer(context), step_size=5000, gamma=0.1)

        handlers = [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
            TensorBoardStatsHandler(
                log_dir=context.events_dir,
                tag_name="train_loss",
                output_transform=from_engine(["loss"], first=True),
            ),
        ]

        if context.evaluator:
            logger.info(
                f"{context.local_rank} - Adding Validation Handler to run every '{self._val_interval}' interval"
            )
            handlers.append(
                ValidationHandler(validator=context.evaluator, interval=self._val_interval, epoch_level=True)
            )

        return (
            handlers if context.local_rank == 0 else [handlers[0], handlers[-1]] if context.evaluator else handlers[:1]
        )

    def train_additional_metrics(self, context: Context):
        return None

    def val_data_loader(self, context: Context, num_workers=0):
        dataset, datalist = self._dataset(context, context.val_datalist)
        logger.info(f"{context.local_rank} - Records for Validation: {len(datalist)}")
        logger.debug(f"{context.local_rank} - Validation: {datalist}")

        return self._dataloader(context, dataset, context.val_batch_size, num_workers)

    def val_pre_transforms(self, context: Context):
        return self.train_pre_transforms(context)

    def val_post_transforms(self, context: Context):
        return self.train_post_transforms(context)

    def val_handlers(self, context: Context):
        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=context.events_dir, output_transform=lambda x: None),
        ]
        return val_handlers if context.local_rank == 0 else None

    def val_key_metric(self, context):
        return {self.VAL_KEY_METRIC: MeanDice(output_transform=from_engine(["pred", "label"]))}

    def train_iteration_update(self, context: Context):
        return None

    def val_iteration_update(self, context: Context):
        return None

    def event_names(self, context: Context):
        return None

    def val_additional_metrics(self, context: Context):
        return None

    @abstractmethod
    def train_pre_transforms(self, context: Context):
        pass

    @abstractmethod
    def train_post_transforms(self, context: Context):
        pass

    @abstractmethod
    def val_inferer(self, context: Context):
        pass

    def partition_datalist(self, context: Context, shuffle=False):
        val_split = context.request.get("val_split", 0.0)
        if val_split > 0.0:
            train_datalist, val_datalist = partition_dataset(
                context.datalist, ratios=[(1 - val_split), val_split], shuffle=shuffle
            )
        else:
            train_datalist = context.datalist
            val_datalist = []

        if context.local_rank == 0:
            logger.info(f"Total Records for Training: {len(train_datalist)}")
            logger.info(f"Total Records for Validation: {len(val_datalist)}")
        return train_datalist, val_datalist

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
        logger.info(f"Train Request (input): {request}")

        req = copy.deepcopy(self._config)
        req.update(copy.deepcopy(request))
        req["run_id"] = datetime.now().strftime("%Y%m%d_%H%M")

        multi_gpu = req["multi_gpu"]
        multi_gpus = req.get("gpus", "all")
        world_size = torch.cuda.device_count() if not multi_gpus or multi_gpus == "all" else len(multi_gpus.split(","))

        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        datalist = self.pre_process(req, datastore)

        if multi_gpu and world_size < 2:
            logger.info("Distributed/Multi GPU is limited")
            multi_gpu = False
            req["multi_gpu"] = False

        if multi_gpu:
            logger.info("Distributed/Multi GPU Training = TRUE")
            tfile = tempfile.NamedTemporaryFile().name
            if any(platform.win32_ver()):
                req["distributed_backend"] = "gloo"
                req["distributed_url"] = f"file://{tfile}"
            torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, req, datalist, self))
            remove_file(tfile)
        else:
            logger.info("Distributed Training = FALSE")
            return self.train(0, world_size, req, datalist)

        self.cleanup(req)
        if os.path.exists(self._stats_path):
            with open(self._stats_path) as f:
                return json.load(f)
        return {}

    def train(self, rank, world_size, request, datalist):
        start_ts = time.time()

        context: Context = Context()

        context.start_ts = start_ts
        context.request = request
        context.datalist = datalist
        context.local_rank = rank
        context.world_size = world_size

        context.run_id = request["run_id"]
        context.multi_gpu = request["multi_gpu"]
        if context.multi_gpu:
            os.environ["LOCAL_RANK"] = str(context.local_rank)

        logger.info(f"{context.local_rank} - Train Request (final): {request}")

        context.device = self._device(context)
        context.max_epochs = request["max_epochs"]
        context.train_batch_size = request["train_batch_size"]
        context.val_batch_size = request["val_batch_size"]
        context.pretrained = request["pretrained"]
        context.dataset_type = request["dataset"]
        context.dataloader_type = request["dataloader"]

        context.output_dir = os.path.join(self._model_dir, request["name"])
        context.events_dir = os.path.join(context.output_dir, f"events_{context.run_id}")

        if not os.path.exists(context.output_dir):
            os.makedirs(context.output_dir, exist_ok=True)

        context.train_datalist, context.val_datalist = self.partition_datalist(context)
        context.network = self._create_network(context)
        context.evaluator = self._create_evaluator(context)
        context.trainer = self._create_trainer(context)

        # Finalize and Run Training
        self.finalize(context)
        context.trainer.run()

        if context.multi_gpu:
            torch.distributed.destroy_process_group()

        # Try to clear cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return prepare_stats(start_ts, context.trainer, context.evaluator)

    def finalize(self, context):
        if context.local_rank == 0:
            publisher = PublishStatsAndModel(
                self._stats_path,
                self._publish_path,
                self._key_metric_filename,
                context.start_ts,
                context.run_id,
                context.output_dir,
                context.trainer,
                context.evaluator,
            )
            if context.evaluator:
                context.evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=publisher)
            else:
                context.trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=publisher)

        early_stop_patience = int(context.request.get("early_stop_patience", 0))
        if early_stop_patience > 0 and context.evaluator:
            early_stopper = EarlyStopping(
                patience=early_stop_patience,
                score_function=stopping_fn_from_metric(self.VAL_KEY_METRIC),
                trainer=context.trainer,
            )
            context.evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

    def pre_process(self, request, datastore: Datastore):
        return datastore.datalist()

    def cleanup(self, request):
        pass

    def _device(self, context: Context):
        if context.multi_gpu:
            distributed_backend = context.request.get("distributed_backend", "nccl")
            distributed_url = context.request.get("distributed_url", "env://")
            torch.distributed.init_process_group(
                backend=distributed_backend,
                init_method=distributed_url,
                world_size=context.world_size,
                rank=context.local_rank,
            )

            gpus = context.request.get("gpus", "all")
            multi_gpus = list(range(context.world_size)) if gpus == "all" else [int(g) for g in gpus.split(",")]
            gpu = multi_gpus[context.local_rank]

            logger.info(f"++++ Rank:{context.local_rank} => Using GPU-{gpu}")
            device = torch.device("cuda:{}".format(gpu))
            torch.cuda.set_device(device)
        else:
            device = torch.device(context.request["device"] if torch.cuda.is_available() else "cpu")

        logger.info(f"{context.local_rank} - Using Device: {device}; IDX: {device.index}")
        return device

    def _create_network(self, context: Context):
        network = self.network(context).to(context.device)
        if context.multi_gpu:
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[context.device.index],
                output_device=context.device.index,
                find_unused_parameters=self._find_unused_parameters,
            )
        return network

    def _create_evaluator(self, context: Context):
        evaluator = None
        if context.val_datalist and len(context.val_datalist) > 0:
            val_hanlders: List = self.val_handlers(context)
            if context.local_rank == 0:
                val_hanlders.append(
                    CheckpointSaver(
                        save_dir=context.output_dir,
                        save_dict={self._model_dict_key: context.network},
                        save_key_metric=True,
                        key_metric_filename=self._key_metric_filename,
                    )
                )

            evaluator = SupervisedEvaluator(
                device=context.device,
                val_data_loader=self.val_data_loader(context),
                network=context.network,
                inferer=self.val_inferer(context),
                postprocessing=self._validate_transforms(self.val_post_transforms(context), "Validation", "post"),
                key_val_metric=self.val_key_metric(context),
                additional_metrics=self.val_additional_metrics(context),
                val_handlers=val_hanlders,
                iteration_update=self.val_iteration_update(context),
                event_names=self.event_names(context),
            )
        return evaluator

    def _create_trainer(self, context: Context):
        train_handlers: List = self.train_handlers(context)
        if context.local_rank == 0:
            train_handlers.append(
                CheckpointSaver(
                    save_dir=context.output_dir,
                    save_dict={self._model_dict_key: context.network},
                    save_interval=self._train_save_interval,
                    save_final=True,
                    final_filename=self._final_filename,
                    save_key_metric=True,
                    key_metric_filename=f"train_{self._key_metric_filename}"
                    if context.evaluator
                    else self._key_metric_filename,
                )
            )

        self._load_checkpoint(context, train_handlers)

        return SupervisedTrainer(
            device=context.device,
            max_epochs=context.max_epochs,
            train_data_loader=self.train_data_loader(context),
            network=context.network,
            optimizer=self.optimizer(context),
            loss_function=self.loss_function(context),
            inferer=self.train_inferer(context),
            amp=self._amp,
            postprocessing=self._validate_transforms(self.train_post_transforms(context), "Training", "post"),
            key_train_metric=self.train_key_metric(context),
            train_handlers=train_handlers,
            iteration_update=self.train_iteration_update(context),
            event_names=self.event_names(context),
        )

    def _load_checkpoint(self, context, train_handlers):
        load_path = self.load_path(context.output_dir, context.pretrained)
        if load_path and os.path.exists(load_path):
            logger.info(f"{context.local_rank} - Load Path {load_path}")

            load_dict = {self._model_dict_key: context.network} if self._load_dict is None else self._load_dict
            map_location = {"cuda:0": "cuda:{}".format(context.device.index)} if context.multi_gpu else None
            train_handlers.append(
                CheckpointLoader(load_path, load_dict, map_location=map_location, strict=self._load_strict)
            )


def main_worker(rank, world_size, request, datastore: Datastore, task: BasicTrainTask):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(f"Main Worker: {rank}")
    task.train(rank, world_size, request, datastore)
