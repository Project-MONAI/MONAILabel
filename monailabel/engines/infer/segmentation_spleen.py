from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    Activationsd,
    AsDiscreted,
    SqueezeDimd,
    ToNumpyd
)

from monailabel.interface import InferenceEngine
from monailabel.interface.utils import Restored, ExtremePointsd, BoundingBoxd


class SegmentationSpleen(InferenceEngine):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
    """

    def __init__(self, model):
        super().__init__(model=model)

    def pre_transforms(self):
        return [
            LoadImaged(keys='image'),
            AddChanneld(keys='image'),
            Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0]),
            ScaleIntensityRanged(keys='image', a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=[160, 160, 160])

    def post_transforms(self):
        return [
            AddChanneld(keys='pred'),
            Activationsd(keys='pred', softmax=True),
            AsDiscreted(keys='pred', argmax=True),
            SqueezeDimd(keys='pred', dim=0),
            ToNumpyd(keys='pred'),
            Restored(keys='pred', ref_image='image'),
            ExtremePointsd(keys='pred', result='result', points='points'),
            BoundingBoxd(keys='pred', result='result', bbox='bbox'),
        ]
