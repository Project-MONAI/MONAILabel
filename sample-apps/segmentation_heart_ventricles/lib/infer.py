from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import Restored


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained heart ventricles segmentation (DynUNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=("left ventricular volume", "left ventricle wall", "right ventricle of heart"),
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of heart from MR image",
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
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(
                keys="image",
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            Orientationd(keys="image", axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            CenterSpatialCropd(keys="image", roi_size=[160, 160, 160]),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=[128, 128, 128])

    def inverse_transforms(self):
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self):
        return [
            ToTensord(keys=("image", "pred")),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
