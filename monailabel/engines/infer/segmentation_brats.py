from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    AddChanneld,
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interface import InferenceEngine, InferType
from monailabel.interface.utils import Restored, BoundingBoxd


class InferSegmentationBrats(InferenceEngine):
    """
    This provides Inference Engine for pre-trained brain tumour segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.SEGMENTATION,
            labels=["brain"],
            dimension=3,
            description='A pre-trained model for volumetric (3D) segmentation of brain tumour over 3D MR Images'
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
        pixdim = (1.5, 1.5, 2.0)
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
        return SlidingWindowInferer(roi_size=[128, 128, 64])

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
