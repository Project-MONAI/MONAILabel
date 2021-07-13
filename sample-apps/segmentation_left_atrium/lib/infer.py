from monai.inferers import SimpleInferer
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
    This provides Inference Engine for pre-trained left atrium segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="left_atrium",
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of the left atrium over 3D MR Images",
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
        pre_transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys="image"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            CenterSpatialCropd(keys="image", roi_size=(256, 256, 128)),
            ToTensord(keys=["image"]),
        ]
        return pre_transforms

    def inferer(self):
        return SimpleInferer()

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
