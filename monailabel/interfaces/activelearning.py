import logging
import random

from monailabel.interfaces.datastore import Datastore

logger = logging.getLogger(__name__)


class ActiveLearning:

    def __call__(self, request, datastore: Datastore):

        strategy = request.get('strategy', 'random')
        strategy = next(iter(strategy)) if not isinstance(strategy, str) else strategy

        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        images.sort()
        if strategy == "first":
            image = images[0]
        elif strategy == "last":
            image = images[-1]
        else:
            image = random.choice(images)

        logger.info(f"Strategy: {strategy}; Selected Image: {image}")
        return image
