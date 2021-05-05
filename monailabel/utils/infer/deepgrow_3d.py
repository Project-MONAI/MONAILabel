from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    SpatialCropGuidanced,
    ResizeGuidanced,
    AddGuidanceSignald,
    RestoreLabeld
)
from monai.inferers import SimpleInferer
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    LoadImaged,
    Compose,
    AsChannelFirstd,
    AddChanneld,
    Spacingd,
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    ToNumpyd,
    Resized,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    AsChannelLastd,
    ToTensord,
    SqueezeDimd,
)
from monailabel.utils.others.post import Restored, BoundingBoxd
from monailabel.utils.infer import InferenceTask, InferType


class InferDeepgrow3D(InferenceTask):
    """
    This provides Inference Engine for Deepgrow-3D pre-trained model.
    For More Details, Refer https://github.com/Project-MONAI/tutorials/tree/master/deepgrow/ignite
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.DEEPGROW,
            labels=[],
            dimension=3,
            description='A pre-trained 3D DeepGrow model based on UNET',
            spatial_size=[128, 128],
            model_size=[128, 128, 128]
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description
        )

        self.spatial_size = spatial_size
        self.model_size = model_size

    def pre_transforms(self):
        # Working for heart images
        return [
            LoadImaged(keys='image'),
            AsChannelFirstd(keys='image'),
            Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0], mode='bilinear'),
            AddGuidanceFromPointsd(ref_image='image', guidance='guidance', dimensions=3),
            AddChanneld(keys='image'),
            SpatialCropGuidanced(keys='image', guidance='guidance', spatial_size=self.spatial_size),
            Resized(keys='image', spatial_size=self.model_size, mode='area'),
            ResizeGuidanced(guidance='guidance', ref_image='image'),
            NormalizeIntensityd(keys='image'),
            AddGuidanceSignald(image='image', guidance='guidance'),
        ]

        # # Testing for DeepEdit in Spleen - STILL CHECKING WHETHER THESE TRANSFORMS MAKE SENSE
        # return [
        #     LoadImaged(keys='image'),
        #     Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0], mode='bilinear'),
        #     Orientationd(keys=["image"], axcodes="RAS"),
        #     NormalizeIntensityd(keys='image'),
        #     AddGuidanceFromPointsd(ref_image='image', guidance='guidance', dimensions=3),
        #     AddChanneld(keys=('image')),
        #     CropForegroundd(keys=('image'), source_key='image', select_fn=lambda x: x > 1.5, margin=3),
        #     Resized(keys='image', spatial_size=(96,96,96), mode='area'),
        #     AddGuidanceSignald(image='image', guidance='guidance'),
        #     ToTensord(keys=('image'))
        # ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        # # Working for heart
        return [
            Activationsd(keys='pred', sigmoid=True),
            AsDiscreted(keys='pred', threshold_values=True, logit_thresh=0.51),
            ToNumpyd(keys='pred'),
            RestoreLabeld(keys='pred', ref_image='image', mode='nearest'),
            AsChannelLastd(keys='pred')
        ]

        # return [
        #     Activationsd(keys='pred', sigmoid=True),
        #     AsDiscreted(keys='pred', threshold_values=True, logit_thresh=0.51),
        #     SqueezeDimd(keys='pred', dim=0),
        #     ToNumpyd(keys='pred'),
        #     Restored(keys='pred', ref_image='image'),
        #     BoundingBoxd(keys='pred', result='result', bbox='bbox'),
        # ]