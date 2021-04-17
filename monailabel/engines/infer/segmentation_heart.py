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

from monailabel.interface import InferenceEngine, InferType
from monailabel.interface.utils import Restored, BoundingBoxd


class SegmentationHeart(InferenceEngine):
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
        return [
            LoadImaged(keys='image'),
            AddChanneld(keys='image'),
            Spacingd(keys='image', pixdim=[0.79, 0.79, 1.24]),
            ScaleIntensityRanged(keys='image', a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ]

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
