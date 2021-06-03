import logging

import numpy as np
from monai.transforms import LoadImage

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.tasks import ScoringMethod

logger = logging.getLogger(__name__)


class Dice(ScoringMethod):
    """
    Compute dice between final vs original tags
    """

    def __init__(self):
        super().__init__("Compute Dice for predicated label vs submitted")

    def __call__(self, request, datastore: Datastore):
        loader = LoadImage(image_only=True)

        tag_y = request.get("y", DefaultLabelTag.FINAL)
        tag_y_pred = request.get("y_pred", DefaultLabelTag.ORIGINAL)

        result = {}
        for image_id in datastore.list_images():
            y_i = datastore.get_label_by_image_id(image_id, tag_y) if tag_y else None
            y_pred_i = datastore.get_label_by_image_id(image_id, tag_y_pred) if tag_y_pred else None

            if y_i and y_pred_i:
                y = loader(datastore.get_label_uri(y_i))
                y_pred = loader(datastore.get_label_uri(y_pred_i))

                y = y.flatten()
                y_pred = y_pred.flatten()
                union = np.sum(y) + np.sum(y_pred)
                dice = 2.0 * np.sum(y * y_pred) / union if union != 0 else 1

                logger.info(f"Dice Score for {image_id} is {dice}")
                datastore.update_image_info(image_id, {"dice": dice})
                result[image_id] = dice
        return result
