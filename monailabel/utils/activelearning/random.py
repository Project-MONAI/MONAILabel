import logging
import random

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks import Strategy

logger = logging.getLogger(__name__)


class Random(Strategy):
    """
    Consider implementing a random strategy for active learning
    """

    def __init__(self):
        super().__init__("Random Strategy")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        image_id = random.choice(list(images.keys()))
        image_path = images[image_id]

        logger.info(f"Random: Selected Image: {image_id}")
        return image_id, image_path
