import numpy as np
import monai
from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    Compose,
    AddChanneld,
    AsChannelLastd,
    AsDiscreted,
    CastToTyped,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    Spacingd,
    SpatialPadd,
    SqueezeDimd,
    Spacingd,
    ToNumpyd,
    ToTensord,
)
from monai.transforms.transform import Transform
from monailabel.utils.infer import InferenceTask, InferType
from monailabel.utils.others.post import Restored, BoundingBoxd

# Create channels with positive and negative points in zero
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
        signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        d[self.image] = np.concatenate((image, signal, signal), axis=0)
        # # Save input image
        # saver = monai.data.NiftiSaver(output_dir='/home/adp20local/Documents/MONAILabel/sample-apps/segmentation_spleen/', mode="nearest")
        # saver.save(d[self.image], meta_data={'affine': np.eye(4), 'filename_or_obj': 'image'})
        return d

class MyInfer(InferenceTask):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.SEGMENTATION,
            labels=["spleen"],
            dimension=3,
            description='A pre-trained model for volumetric (3D) segmentation of the spleen from CT image'
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
        pre_transforms = Compose([
                        LoadImaged(keys=('image')),
                        AddChanneld(keys=('image')),
                        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                        Orientationd(keys=["image"], axcodes="RAS"),
                        NormalizeIntensityd(keys='image'),
                        CropForegroundd(keys=('image'), source_key='image', select_fn=lambda x: x > x.max()*0.6, margin=3), # select_fn and margin are Task dependant
                        Resized(keys=('image'), spatial_size=(96,96,96), mode=('area')),
                        DiscardAddGuidanced(image='image'),
                        ToTensord(keys=('image'))
                        ])
        return pre_transforms

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            ToTensord(keys=('image')),
            Activationsd(keys='image', sigmoid=True),
            AsDiscreted(keys='image', threshold_values=True, logit_thresh=0.51),
            SqueezeDimd(keys='image', dim=0),
            ToNumpyd(keys='image'),
        ]