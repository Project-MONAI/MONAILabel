from monai.apps.deepgrow.transforms import AddGuidanceFromPointsd, AddGuidanceSignald
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    Spacingd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.transforms import DiscardAddGuidanced, ResizeGuidanceCustomd
from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import Restored


class Segmentation(InferTask):
    """
    This provides Inference Engine for pre-trained left atrium segmentation (Dynamic UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="left atrium",
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
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            AddChanneld(keys="image"),
            Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys="image", axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            Resized(keys="image", spatial_size=(256, 256, 128), mode="area"),
            DiscardAddGuidanced(image="image"),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SimpleInferer()

    def inverse_transforms(self):
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self):
        return [
            ToTensord(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.51),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]


class Deepgrow(InferTask):
    """
    This provides Inference Engine for Deepgrow over DeepEdit model.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPGROW,
        dimension=3,
        description="A pre-trained 3D DeepGrow model based on UNET",
        spatial_size=(256, 256),
        model_size=(256, 256, 128),
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=None,
            dimension=dimension,
            description=description,
        )

        self.spatial_size = spatial_size
        self.model_size = model_size

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            AddChanneld(keys="image"),
            Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys="image", axcodes="RAS"),
            SqueezeDimd(keys="image", dim=0),
            AddGuidanceFromPointsd(ref_image="image", guidance="guidance", dimensions=3),
            AddChanneld(keys="image"),
            NormalizeIntensityd(keys="image"),
            Resized(keys="image", spatial_size=self.model_size, mode="area"),
            ResizeGuidanceCustomd(guidance="guidance", ref_image="image"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            ToTensord(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.51),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
