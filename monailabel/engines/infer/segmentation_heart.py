import numpy as np
from typing import Dict

from monai.inferers import SlidingWindowInferer
from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    SpatialCropGuidanced,
    ResizeGuidanced,
    AddGuidanceSignald,
    RestoreLabeld
)
from monai.transforms import (
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    Spacingd,
    Activationsd,
    AsDiscreted,
    ToNumpyd,
    Resized,
    NormalizeIntensityd,
    AsChannelLastd
)
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interface import InferenceEngine, InferType
from monailabel.interface.utils import Restored, BoundingBoxd
from monai.transforms.transform import MapTransform, Randomizable, Transform

# Define a new transform to discard positive and negative points
class DiscardAddGuidanced(Transform):
    """
    Discard positive and negative points randomly or Add the two channels for inference time
    """
    def __init__(self, image: str = "image", batched: bool = False,):
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
        # pixdim = (0.79, 0.79, 1.24)
        # roi_size = [192, 160, 80]
        # pre_transforms = [
        #     LoadImaged(keys=["image"]),
        #     AddChanneld(keys=["image"]),
        #     Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear", ),
        #     Orientationd(keys=["image"], axcodes="RAS"),
        #     SpatialPadd(keys=["image"], spatial_size=tuple(roi_size)),
        #     NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        #     CastToTyped(keys=["image"], dtype=np.float32),
        #     ToTensord(keys=["image"]),
        # ]
        # return pre_transforms

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

