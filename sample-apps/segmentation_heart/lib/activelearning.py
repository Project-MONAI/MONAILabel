import logging

from monailabel.interfaces import Datastore
from monailabel.interfaces.tasks import Strategy

logger = logging.getLogger(__name__)


class MyStrategy(Strategy):
    """
    Consider implementing a first strategy for active learning
    """

    def __init__(self):
        super().__init__("Get First Sample")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        images.sort()
        image = images[0]

        logger.info(f"First: Selected Image: {image}")
        return image
