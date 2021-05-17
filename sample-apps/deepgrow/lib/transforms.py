import logging
from typing import Dict

import numpy as np
from monai.transforms import RandomizableTransform

logger = logging.getLogger(__name__)


# You can write your transforms here... which can be used in your train/infer tasks
class Random2DSlice(RandomizableTransform):
    def __init__(self, image: str = "image", label: str = "label"):
        super().__init__()

        self.image = image
        self.label = label

    def __call__(self, data):
        d: Dict = dict(data)
        image = d[self.image]
        label = d[self.label]

        if len(image.shape) and len(label.shape) != 3:  # only for 3D
            raise ValueError("Only supports label with shape DHW!")

        sids = []
        for sid in range(label.shape[0]):
            if np.sum(label[sid]) != 0:
                sids.append(sid)

        sid = self.R.choice(sids, replace=False)
        d[self.image] = image[sid]
        d[self.label] = label[sid]
        return d
