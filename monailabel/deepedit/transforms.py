import logging
from typing import Dict

import numpy as np
from monai.transforms import Transform

logger = logging.getLogger(__name__)


class DiscardAddGuidanced(Transform):
    def __init__(self, image: str = "image", probability: float = 1.0):
        """
        Discard positive and negative points randomly or Add the two channels for inference time

        :param image: image key
        :param batched: Is it batched (if used during training and data is batched as interaction transform)
        :param probability: Discard probability; For inference it will be always 1.0
        """
        self.image = image
        self.probability = probability

    def _apply(self, image):
        if self.probability >= 1.0 or np.random.choice([True, False], p=[self.probability, 1 - self.probability]):
            signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            if image.shape[0] == 3:
                image[1] = signal
                image[2] = signal
            else:
                image = np.concatenate((image, signal, signal), axis=0)
        return image

    def __call__(self, data):
        d: Dict = dict(data)
        d[self.image] = self._apply(d[self.image])
        return d


class ResizeGuidanceCustomd(Transform):
    """
    Resize the guidance based on cropped vs resized image.
    """

    def __init__(
        self,
        guidance: str,
        ref_image: str,
    ) -> None:
        self.guidance = guidance
        self.ref_image = ref_image

    def __call__(self, data):
        d = dict(data)
        current_shape = d[self.ref_image].shape[1:]

        factor = np.divide(current_shape, d["image_meta_dict"]["dim"][1:4])
        pos_clicks, neg_clicks = d["foreground"], d["background"]

        pos = np.multiply(pos_clicks, factor).astype(int).tolist() if len(pos_clicks) else []
        neg = np.multiply(neg_clicks, factor).astype(int).tolist() if len(neg_clicks) else []

        d[self.guidance] = [pos, neg]
        return d
