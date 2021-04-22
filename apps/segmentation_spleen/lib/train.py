import logging

from monailabel.engines.train import TrainSegmentation

logger = logging.getLogger(__name__)


class MyTrain(TrainSegmentation):
    def train_pre_transforms(self):
        pass

    def train_post_transforms(self):
        pass

    def val_inferer(self):
        pass
