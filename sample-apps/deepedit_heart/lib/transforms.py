from typing import Dict

import numpy as np

from monai.transforms import Transform


class AddEmptyGuidanced(Transform):
    """
    Create channels with positive and negative points in zero. Add empty two channels for inference time
    """

    def __init__(self, image: str = "image"):
        self.image = image

    def __call__(self, data):
        d: Dict = dict(data)
        image = d[self.image]

        # For pure inference time - There is no positive neither negative points
        signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        d[self.image] = np.concatenate((image, signal, signal), axis=0)
        return d
