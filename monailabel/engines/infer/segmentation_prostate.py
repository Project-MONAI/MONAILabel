from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    AddChanneld,
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    Spacingd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
    NormalizeIntensityd,
    Orientationd,
)

from monailabel.interface import InferenceEngine, InferType
from monailabel.interface.utils import Restored, BoundingBoxd


class InferSegmentationProstate(InferenceEngine):
    """
    This provides Inference Engine for pre-trained prostate segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.SEGMENTATION,
            labels=["prostate"],
            dimension=3,
            description='A pre-trained model for volumetric (3D) segmentation of the prostate over 3D MR Images'
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
        pixdim = (0.62, 0.62, 3.6)
        pre_transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
        return pre_transforms

    def inferer(self):
        return SlidingWindowInferer(roi_size=[320, 256, 20])

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
