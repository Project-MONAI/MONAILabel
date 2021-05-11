from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropGuidanced,
)
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsChannelLastd,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    Spacingd,
    ToNumpyd,
)

from monailabel.interfaces.tasks import InferTask, InferType


class InferDeepgrow3D(InferTask):
    """
    This provides Inference Engine for Deepgrow-3D pre-trained model.
    For More Details, Refer https://github.com/Project-MONAI/tutorials/tree/master/deepgrow/ignite
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPGROW,
        labels=None,
        dimension=3,
        description="A pre-trained 2D DeepGrow model based on UNET",
        spatial_size=(256, 256),
        model_size=(128, 192, 192),
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

        self.spatial_size = spatial_size
        self.model_size = model_size

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            AsChannelFirstd(keys="image"),
            Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0], mode="bilinear"),
            AddGuidanceFromPointsd(ref_image="image", guidance="guidance", dimensions=3),
            AddChanneld(keys="image"),
            SpatialCropGuidanced(keys="image", guidance="guidance", spatial_size=self.spatial_size),
            Resized(keys="image", spatial_size=self.model_size, mode="area"),
            ResizeGuidanced(guidance="guidance", ref_image="image"),
            NormalizeIntensityd(keys="image", subtrahend=208, divisor=388),
            AddGuidanceSignald(image="image", guidance="guidance"),
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ToNumpyd(keys="pred"),
            RestoreLabeld(keys="pred", ref_image="image", mode="nearest"),
            AsChannelLastd(keys="pred"),
        ]
