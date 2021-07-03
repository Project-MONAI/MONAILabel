from monai.inferers import SlidingWindowInferer
from monai.transforms import (  # CenterSpatialCropd,
    Activationsd,
    AddChanneld,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    Spacingd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import BoundingBoxd, Restored


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained whole heart segmentation (DynUNet) model.
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
        pixdim = (1.0, 1.0, 1.0)
        roi_size = [128, 128, 128]
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(
                keys="image",
                pixdim=pixdim,
                mode="bilinear",
            ),
            Orientationd(keys="image", axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            # CenterSpatialCropd(keys="image", roi_size=[128, 128, 128]),
            Resized(keys="image", spatial_size=roi_size, mode="area"),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=[128, 128, 128])

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
