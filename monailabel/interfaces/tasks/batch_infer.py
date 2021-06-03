import copy
import logging
from enum import Enum
from typing import Callable

from monailabel.interfaces.datastore import Datastore

logger = logging.getLogger(__name__)


class BatchInferImageType(str, Enum):
    IMAGES_ALL = "all"
    IMAGES_LABELED = "labeled"
    IMAGES_UNLABELED = "unlabeled"


class BatchInferTask:
    """
    Basic Batch Infer Task
    """

    def get_images(self, request, datastore: Datastore):
        """
        Override this method to get all eligible images for your task to run batch infer
        """
        images = request.get("images", BatchInferImageType.IMAGES_ALL)
        if isinstance(images, str):
            if images == BatchInferImageType.IMAGES_LABELED:
                return datastore.get_labeled_images()
            if images == BatchInferImageType.IMAGES_UNLABELED:
                return datastore.get_unlabeled_images()
            return datastore.list_images()
        return images

    def __call__(self, request, datastore: Datastore, infer: Callable):
        image_ids = self.get_images(request, datastore)
        logger.info(f"Total number of images for batch inference: {len(image_ids)}")

        result = {}
        for image_id in image_ids:
            req = copy.deepcopy(request)
            req["image"] = image_id

            logger.info(f"Running inference for image id {image_id}")
            result[image_id] = infer(req, datastore)
        return result
