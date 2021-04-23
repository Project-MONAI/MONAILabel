from typing import Dict

import numpy as np

from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Spacingd,
    SqueezeDimd,
    ToNumpyd,
)
from monai.transforms import (
    AsChannelFirstd,
    Resized
)
from monai.transforms.transform import Transform
from monailabel.helpers.infer import InferenceTask, InferType
from monailabel.helpers.others import Restored, BoundingBoxd


# Define a new transform to discard positive and negative points
class DiscardAddGuidanced(Transform):
    """
    Discard positive and negative points randomly or Add the two channels for inference time
    """

    def __init__(self, image: str = "image", batched: bool = False, ):
        self.image = image
        # What batched means/implies? I see that the dictionary is in the list form instead of numpy array
        self.batched = batched

    def __call__(self, data):
        d: Dict = dict(data)
        image = d[self.image]
        # For pure inference time - There is no positive neither negative points
        print('This is the image shape: ', image.shape)
        signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        d[self.image] = np.concatenate((d[self.image], signal, signal), axis=0)
        print('This is the output image shape: ', d[self.image].shape)
        return d


class InferDeepEdit(InferenceTask):
    """
    This provides Inference Engine for pre-trained heart segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.SEGMENTATION,
            labels=["heart"],
            dimension=3,
            description='A pre-trained model for volumetric (3D) segmentation of the heart over 3D MR Images'
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description
        )

    def pre_transforms(self):
        pre_transforms = [
            LoadImaged(keys='image'),
            AsChannelFirstd(keys='image'),
            Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0], mode='bilinear'),
            # AddGuidanceFromPointsd(ref_image='image', guidance='guidance', dimensions=3),
            AddChanneld(keys='image'),
            # SpatialCropGuidanced(keys='image', guidance='guidance', spatial_size=[192, 160, 80]),
            Resized(keys='image', spatial_size=[192, 160, 80], mode='area'),
            # ResizeGuidanced(guidance='guidance', ref_image='image'),
            NormalizeIntensityd(keys='image', subtrahend=208, divisor=388),
            # AddGuidanceSignald(image='image', guidance='guidance'),
            DiscardAddGuidanced(image='image'),
        ]
        return pre_transforms

    def inferer(self):
        return SlidingWindowInferer(roi_size=[192, 160, 80])

    def post_transforms(self):
        return [
            AddChanneld(keys='pred'),
            Activationsd(keys='pred', softmax=True),
            AsDiscreted(keys='pred', argmax=True),
            SqueezeDimd(keys='pred', dim=0),
            ToNumpyd(keys='pred'),
            Restored(keys='pred', ref_image='image'),
            BoundingBoxd(keys='pred', result='result', bbox='bbox'),
        ]
