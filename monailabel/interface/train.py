import datetime
import logging
import time
from abc import abstractmethod

from monai.engines import SupervisedTrainer, SupervisedEvaluator

logger = logging.getLogger(__name__)


# TODO:: Think of some better abstraction/generalization here... few abstracts can be removed
#  And support Multi GPU
class TrainEngine(object):

    @abstractmethod
    def device(self):
        """
        Provide device name
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
        return None

    def evaluator(self):
        return SupervisedEvaluator(
            device=self.device(),
            val_data_loader=self.val_data_loader(),
            network=self.network().to(self.device()),
            inferer=self.val_inferer(),
            post_transform=self.val_post_transforms(),
            key_val_metric=self.val_key_metric(),
            additional_metrics=self.val_additional_metrics(),
            val_handlers=self.val_handlers()
        )

    def run(self, max_epochs, amp):
        trainer = SupervisedTrainer(
            device=self.device(),
            max_epochs=max_epochs,
            train_data_loader=self.train_data_loader(),
            network=self.network().to(self.device()),
            optimizer=self.optimizer(),
            loss_function=self.loss_function(),
            inferer=self.train_inferer(),
            amp=amp,
            post_transform=self.train_post_transforms(),
            key_train_metric=self.train_key_metric(),
            train_handlers=self.train_handlers(),
        )

        logger.info(f"Running Training.  Epochs: {max_epochs}; AMP: {amp}")
        start = time.time()
        trainer.run()
        lapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
        logger.info(f"++ Total Train Time:: {lapsed}")

        stats = trainer.get_train_stats()
        stats['total_time'] = lapsed
        logger.info(f"Train Stats: {stats}")
        return stats
