import datetime
import logging
import time
from abc import abstractmethod
from typing import Any, Dict

import torch
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.transforms.compose import Compose

logger = logging.getLogger(__name__)


class TrainTask:
    def __init__(self):
        self._start_time = time.time()
        self._trainer = None
        self._evaluator = None

    @abstractmethod
    def device(self):
        """
        Provide device name
        """
        pass

    @abstractmethod
    def max_epochs(self):
        """
        Max number of epochs to run
        """
        pass

    @abstractmethod
    def amp(self):
        """
        Use AMP
        """
        pass

    @abstractmethod
    def network(self):
        """
        Provide network
        """
        pass

    @abstractmethod
    def loss_function(self):
        """
        Provide Lost function
        """
        pass

    @abstractmethod
    def optimizer(self):
        """
        Provide Optimizer
        """
        pass

    @abstractmethod
    def train_pre_transforms(self):
        """
        Provide List of Pre-Transforms for training
        """
        pass

    @abstractmethod
    def train_post_transforms(self):
        """
        Provide List of Post-Transforms for training
        """
        pass

    @abstractmethod
    def train_data_loader(self):
        """
        Provide Dataloader for training samples
        """
        pass

    @abstractmethod
    def train_inferer(self):
        """
        Provide Inferer to be used while training
        """
        pass

    @abstractmethod
    def train_key_metric(self):
        """
        Provide List of Key Metrics to be collected while training
        """
        pass

    @abstractmethod
    def train_handlers(self):
        """
        Provide List of training handlers
        """
        pass

    @abstractmethod
    def train_additional_metrics(self):
        """
        Provide any additional metrics to be collected while training
        """
        return None

    @abstractmethod
    def train_iteration_update(self):
        """
        Provide iteration update while training
        """
        pass

    @abstractmethod
    def event_names(self):
        """
        Provide iteration update while training
        """
        pass

    @abstractmethod
    def val_pre_transforms(self):
        """
        Provide List of Pre-Transforms for validation step
        """
        pass

    @abstractmethod
    def val_post_transforms(self):
        """
        Provide List of Post-Transforms for validation step
        """
        pass

    @abstractmethod
    def val_inferer(self):
        """
        Provide Inferer to be used for validation step
        """
        pass

    @abstractmethod
    def val_data_loader(self):
        """
        Provide Dataloader for validation samples
        """
        pass

    @abstractmethod
    def val_handlers(self):
        """
        Provide List of handlers for validation
        """
        pass

    @abstractmethod
    def val_key_metric(self):
        """
        Provide List of Key metrics to be collected during validation
        """
        pass

    @abstractmethod
    def val_additional_metrics(self):
        """
        Provide any additional metrics to be collected while validation
        """
        pass

    @abstractmethod
    def val_iteration_update(self):
        """
        Provide iteration update while validation
        """
        pass

    def evaluator(self):

        if isinstance(self.val_post_transforms(), list):
            val_post_transforms = Compose(self.val_post_transforms())
        elif isinstance(self.val_post_transforms(), Compose):
            val_post_transforms = self.val_post_transforms()
        else:
            raise ValueError("Validation post-transforms are not of `list` or `Compose` type")

        if not self._evaluator and self.val_data_loader():
            self._evaluator = SupervisedEvaluator(
                device=self.device(),
                val_data_loader=self.val_data_loader(),
                network=self.network().to(self.device()),
                inferer=self.val_inferer(),
                postprocessing=val_post_transforms,
                key_val_metric=self.val_key_metric(),
                additional_metrics=self.val_additional_metrics(),
                val_handlers=self.val_handlers(),
                iteration_update=self.val_iteration_update(),
                event_names=self.event_names(),
            )
        return self._evaluator

    def trainer(self):

        if isinstance(self.train_post_transforms(), list):
            train_post_transforms = Compose(self.train_post_transforms())
        elif isinstance(self.train_post_transforms(), Compose):
            train_post_transforms = self.train_post_transforms()
        else:
            raise ValueError("Training post-transforms are not of `list` or `Compose` type")

        if not self._trainer:
            self._trainer = SupervisedTrainer(
                device=self.device(),
                max_epochs=self.max_epochs(),
                train_data_loader=self.train_data_loader(),
                network=self.network().to(self.device()),
                optimizer=self.optimizer(),
                loss_function=self.loss_function(),
                inferer=self.train_inferer(),
                amp=self.amp(),
                postprocessing=train_post_transforms,
                key_train_metric=self.train_key_metric(),
                train_handlers=self.train_handlers(),
                iteration_update=self.train_iteration_update(),
                event_names=self.event_names(),
            )
        return self._trainer

    def __call__(self):
        self._start_time = time.time()
        self.trainer().run()

        # Try to clear cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.prepare_stats()

    @staticmethod
    def tensor_to_list(d):
        r = dict()
        for k, v in d.items():
            r[k] = v.tolist() if torch.is_tensor(v) else v
        return r

    def prepare_stats(self):
        stats: Dict[str, Any] = dict()
        stats.update(self._trainer.get_train_stats())
        stats["epoch"] = self._trainer.state.epoch
        stats["start_ts"] = int(self._start_time)

        if self._trainer.state.epoch == self._trainer.state.max_epochs:
            stats["total_time"] = str(datetime.timedelta(seconds=int(time.time() - self._start_time)))
        else:
            stats["current_time"] = str(datetime.timedelta(seconds=int(time.time() - self._start_time)))

        for k, v in {"train": self._trainer, "eval": self._evaluator}.items():
            if not v:
                continue

            stats["best_metric"] = v.state.best_metric
            stats[k] = {
                "metrics": TrainTask.tensor_to_list(v.state.metrics),
                # "metric_details": tensor_to_list(v.state.metric_details),
                "key_metric_name": v.state.key_metric_name,
                "best_metric": v.state.best_metric,
                "best_metric_epoch": v.state.best_metric_epoch,
            }
        return stats
