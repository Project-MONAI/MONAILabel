import copy
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class BatchInferTask:
    """
    Basic Bach Infer Task
    """

    def __call__(self, request, infer: Callable):
        image_ids = request["infer_images"]
        request.pop("infer_images")

        logger.info(f"Total number of images for batch inference: {len(image_ids)}")

        result = {}
        for image_id in image_ids:
            req = copy.deepcopy(request)
            req["image"] = image_id

            logger.info(f"Running inference for image id {image_id}")

            res_label, res_json = infer(req)
            result[image_id] = {
                "label": res_label,
                "params": res_json,
            }

        return result
