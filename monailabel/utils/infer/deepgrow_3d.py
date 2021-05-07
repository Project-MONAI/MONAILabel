import numpy as np
from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    SpatialCropGuidanced,
    AddGuidanceSignald,
    ResizeGuidanced,
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
    Resize,
)
from monailabel.utils.others.post import Restored, BoundingBoxd
from monailabel.utils.infer import InferenceTask, InferType
from monai.transforms.transform import Transform, MapTransform
# from monai.utils import InterpolateMode, ensure_tuple_rep
# from monai.config import KeysCollection
# from typing import Callable, Dict, Optional, Sequence, Union


class ResizeGuidanceCustomd(Transform):
    """
    Resize the guidance based on cropped vs resized image.
    """

    def __init__(
        self,
        guidance: str,
        ref_image: str,
    ) -> None:
        self.guidance = guidance
        self.ref_image = ref_image

    def __call__(self, data):
        d = dict(data)
        # dict_keys(['model', 'image', 'device',
        #            'foreground', 'background', 'image_path',
        #            'image_meta_dict', 'image_transforms',
        #            'guidance', 'foreground_start_coord', 'foreground_end_coord'])
        current_shape = d[self.ref_image].shape[1:]

        print('foreground: ----', d['foreground'])
        print('background: ----', d['background'])
        print('guidance: -----', d['guidance'])

        factor = np.divide(current_shape, d['image_meta_dict']['dim'][1:4])

        pos_clicks, neg_clicks = d['foreground'], d['background']
        pos = np.multiply(pos_clicks, factor).astype(int).tolist() if len(pos_clicks) else []
        neg = np.multiply(neg_clicks, factor).astype(int).tolist() if len(neg_clicks) else []
        print('Reshape foreground: ----', pos)
        print('Reshape background: ----', neg)

        d[self.guidance] = [pos, neg]

        return d


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

        # # Working for heart images
        # return [
        #     LoadImaged(keys='image'),
        #     AsChannelFirstd(keys='image'),
        #     Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0], mode='bilinear'),
        #     AddGuidanceFromPointsd(ref_image='image', guidance='guidance', dimensions=3),
        #     AddChanneld(keys='image'),
        #     SpatialCropGuidanced(keys='image', guidance='guidance', spatial_size=self.spatial_size),
        #     Resized(keys='image', spatial_size=self.model_size, mode='area'),
        #     ResizeGuidanced(guidance='guidance', ref_image='image'),
        #     NormalizeIntensityd(keys='image'),
        #     AddGuidanceSignald(image='image', guidance='guidance'),
        # ]

        # Working for spleen images
        return Compose([
            LoadImaged(keys='image'),
            # Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0], mode='bilinear'), # The inverse of this transform causes some issues
            Orientationd(keys="image", axcodes="RAS"),
            AddGuidanceFromPointsd(ref_image='image', guidance='guidance', dimensions=3),
            AddChanneld(keys='image'),
            NormalizeIntensityd(keys='image'),
            # CropForegroundd(keys=('image'), source_key='image', select_fn=lambda x: x > 1.3, margin=3), # For Spleen -- NOT NEEDED - ITS DOESN'T CONTRIBUTE
            Resized(keys='image', spatial_size=self.model_size, mode='area'),
            ResizeGuidanceCustomd(guidance='guidance', ref_image='image'),
            AddGuidanceSignald(image='image', guidance='guidance'),
        ])

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        # # Working for heart
        # return [
        #     Activationsd(keys='pred', sigmoid=True),
        #     AsDiscreted(keys='pred', threshold_values=True, logit_thresh=0.51),
        #     ToNumpyd(keys='pred'),
        #     RestoreLabeld(keys='pred', ref_image='image', mode='nearest'),
        #     AsChannelLastd(keys='pred')
        # ]
        # # Working for spleen
        return [
            ToTensord(keys='image'),
            Activationsd(keys='image', sigmoid=True),
            AsDiscreted(keys='image', threshold_values=True, logit_thresh=0.51),
            SqueezeDimd(keys='image', dim=0),
            ToNumpyd(keys='image'),
        ]