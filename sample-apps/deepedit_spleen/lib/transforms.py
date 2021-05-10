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
        signal = np.zeros(
            (1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32
        )
        d[self.image] = np.concatenate((image, signal, signal), axis=0)
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
        # dict_keys(['model', 'image', 'device',
        #            'foreground', 'background', 'image_path',
        #            'image_meta_dict', 'image_transforms',
        #            'guidance', 'foreground_start_coord', 'foreground_end_coord'])
        current_shape = d[self.ref_image].shape[1:]

        print("foreground: ----", d["foreground"])
        print("background: ----", d["background"])
        print("guidance: -----", d["guidance"])

        factor = np.divide(current_shape, d["image_meta_dict"]["dim"][1:4])

        pos_clicks, neg_clicks = d["foreground"], d["background"]
        pos = (
            np.multiply(pos_clicks, factor).astype(int).tolist()
            if len(pos_clicks)
            else []
        )
        neg = (
            np.multiply(neg_clicks, factor).astype(int).tolist()
            if len(neg_clicks)
            else []
        )
        print("Reshape foreground: ----", pos)
        print("Reshape background: ----", neg)

        d[self.guidance] = [pos, neg]

        return d
