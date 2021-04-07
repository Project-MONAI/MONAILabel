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
        pass

    @abstractmethod
    def network(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass

    @abstractmethod
    def optimizer(self):
        pass

    @abstractmethod
    def train_pre_transforms(self):
        pass

    @abstractmethod
    def train_post_transforms(self):
        pass

    @abstractmethod
    def train_data_loader(self):
        pass

    @abstractmethod
    def train_inferer(self):
        pass

    @abstractmethod
    def train_key_metric(self):
        pass

    @abstractmethod
    def train_handlers(self):
        pass

    @abstractmethod
    def train_additional_metrics(self):
        return None

    @abstractmethod
    def val_pre_transforms(self):
        pass

    @abstractmethod
    def val_post_transforms(self):
        pass

    @abstractmethod
    def val_inferer(self):
        pass

    @abstractmethod
    def val_data_loader(self):
        pass

    @abstractmethod
    def val_handlers(self):
        pass

    @abstractmethod
    def val_key_metric(self):
        pass

    @abstractmethod
    def val_additional_metrics(self):
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
