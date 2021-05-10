import numpy as np
from monai.inferers import SlidingWindowInferer
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

from monailabel.utils.infer import InferenceTask, InferType
from monailabel.utils.others.post import BoundingBoxd, Restored


class MyInfer(InferenceTask):
    """
    This provides Inference Engine for pre-trained heart segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="heart",
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of the heart over 3D MR Images",
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        pixdim = (0.79, 0.79, 1.24)
        roi_size = [192, 160, 80]
        pre_transforms = [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode="bilinear",
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            SpatialPadd(keys=["image"], spatial_size=tuple(roi_size)),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            CastToTyped(keys=["image"], dtype=np.float32),
            ToTensord(keys=["image"]),
        ]
        return pre_transforms

    def inferer(self):
        return SlidingWindowInferer(roi_size=[192, 160, 80])

    def post_transforms(self):
        return [
            AddChanneld(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]
