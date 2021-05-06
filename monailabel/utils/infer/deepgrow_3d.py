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
from monai.utils import InterpolateMode, ensure_tuple_rep
from monai.config import KeysCollection
from typing import Callable, Dict, Optional, Sequence, Union


class ResizeGuidanceCustomd(Transform):
    """
    Resize the guidance based on cropped vs resized image.

    This transform assumes that the images have been cropped and resized. And the shape after cropped is store inside
    the meta dict of ref image.

    Args:
        guidance: key to guidance
        ref_image: key to reference image to fetch current and original image details
        meta_key_postfix: use `{ref_image}_{postfix}` to to fetch the meta data according to the key data,
            default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        cropped_shape_key: key that records cropped shape for foreground.
    """

    def __init__(
        self,
        guidance: str,
        ref_image: str,
        meta_key_postfix="meta_dict",
        cropped_shape_key: str = "foreground_cropped_shape",
    ) -> None:
        self.guidance = guidance
        self.ref_image = ref_image
        self.meta_key_postfix = meta_key_postfix
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        # dict_keys(['model', 'image', 'device',
        #            'foreground', 'background', 'image_path',
        #            'image_meta_dict', 'image_transforms',
        #            'guidance', 'foreground_start_coord', 'foreground_end_coord'])
        d = dict(data)
        guidance = d[self.guidance]
        meta_dict: Dict = d[f"{self.ref_image}_{self.meta_key_postfix}"]
        current_shape = d[self.ref_image].shape[1:]
        print('THIS IS THE CURRENT SHAPE 0:', d[self.ref_image].shape)
        print('THIS IS THE CURRENT SHAPE 1:',current_shape)
        # cropped_shape = meta_dict[self.cropped_shape_key][1:]
        print('foreground: ----', d['foreground'])
        print('background: ----', d['background'])
        print('guidance: -----', d['guidance'])
        factor = np.divide(current_shape, d['image_meta_dict']['dim'][1:4]) # cropped_shape)
        print('FACTOR: ', factor)

        pos_clicks, neg_clicks = d['foreground'], d['background'] # guidance[0], guidance[1]
        print('POSITIVE CLICKS: ', pos_clicks)
        print('NEGATIVE CLICKS: ', neg_clicks)
        pos = np.multiply(pos_clicks, factor).astype(int).tolist() if len(pos_clicks) else []
        neg = np.multiply(neg_clicks, factor).astype(int).tolist() if len(neg_clicks) else []
        print('POS: ', pos)
        print('NEG: ', neg)
        d[self.guidance] = [pos, neg]
        return d

class RestoreLabelCustomd(MapTransform):
    """
    Restores label based on the ref image.

    The ref_image is assumed that it went through the following transforms:

        1. Fetch2DSliced (If 2D)
        2. Spacingd
        3. SpatialCropGuidanced
        4. Resized

    And its shape is assumed to be (C, D, H, W)

    This transform tries to undo these operation so that the result label can be overlapped with original volume.
    It does the following operation:

        1. Undo Resized
        2. Undo SpatialCropGuidanced
        3. Undo Spacingd
        4. Undo Fetch2DSliced

    The resulting label is of shape (D, H, W)

    Args:
        keys: keys of the corresponding items to be transformed.
        ref_image: reference image to fetch current and original image details
        slice_only: apply only to an applicable slice, in case of 2D model/prediction
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function for padding. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        meta_key_postfix: use `{ref_image}_{meta_key_postfix}` to to fetch the meta data according to the key data,
            default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        start_coord_key: key that records the start coordinate of spatial bounding box for foreground.
        end_coord_key: key that records the end coordinate of spatial bounding box for foreground.
        original_shape_key: key that records original shape for foreground.
        cropped_shape_key: key that records cropped shape for foreground.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        slice_only: bool = False,
        mode: Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str] = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = "meta_dict",
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ref_image = ref_image
        self.slice_only = slice_only
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_key_postfix = meta_key_postfix
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        meta_dict: Dict = d[f"{self.ref_image}_{self.meta_key_postfix}"]

        # dict_keys(['model', 'image', 'device',
        #            'foreground', 'background',
        #            'image_path', 'image_meta_dict',
        #            'image_transforms', 'guidance', 'pred'])

        for key, mode, align_corners in self.key_iterator(d, self.mode, self.align_corners):
            image = d[key]

            # Undo Resize
            current_shape = image.shape
            cropped_shape = meta_dict[self.cropped_shape_key]
            print('This is the cropped_shape: ', cropped_shape)
            if np.any(np.not_equal(current_shape, cropped_shape)):
                resizer = Resize(spatial_size=cropped_shape[1:], mode=mode)
                image = resizer(image, mode=mode, align_corners=align_corners)

            # Undo Crop
            original_shape = meta_dict[self.original_shape_key]
            result = np.zeros(original_shape, dtype=np.float32)
            box_start = meta_dict[self.start_coord_key]
            box_end = meta_dict[self.end_coord_key]

            spatial_dims = min(len(box_start), len(image.shape[1:]))
            slices = [slice(None)] + [slice(s, e) for s, e in zip(box_start[:spatial_dims], box_end[:spatial_dims])]
            slices = tuple(slices)
            result[slices] = image

            # Undo Spacing
            current_size = result.shape[1:]
            # change spatial_shape from HWD to DHW
            spatial_shape = list(np.roll(meta_dict["spatial_shape"], 1))
            spatial_size = spatial_shape[-len(current_size) :]

            if np.any(np.not_equal(current_size, spatial_size)):
                resizer = Resize(spatial_size=spatial_size, mode=mode)
                result = resizer(result, mode=mode, align_corners=align_corners)

            # Undo Slicing
            slice_idx = meta_dict.get("slice_idx")
            if slice_idx is None or self.slice_only:
                final_result = result if len(result.shape) <= 3 else result[0]
            else:
                slice_idx = meta_dict["slice_idx"][0]
                final_result = np.zeros(tuple(spatial_shape))
                final_result[slice_idx] = result
            d[key] = final_result

            meta = d.get(f"{key}_{self.meta_key_postfix}")
            if meta is None:
                meta = dict()
                d[f"{key}_{self.meta_key_postfix}"] = meta
            meta["slice_idx"] = slice_idx
            meta["affine"] = meta_dict["original_affine"]
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

        # Work in progress -  for spleen images
        return Compose([
            LoadImaged(keys='image'),
            # AsChannelFirstd(keys='image'),
            Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0], mode='bilinear'),
            Orientationd(keys=["image"], axcodes="RAS"),
            AddGuidanceFromPointsd(ref_image='image', guidance='guidance', dimensions=3),
            AddChanneld(keys='image'),
            NormalizeIntensityd(keys='image'),
            # SpatialCropGuidanced(keys='image', guidance='guidance', spatial_size=self.spatial_size),
            # CropForegroundd(keys=('image'), source_key='image', select_fn=lambda x: x > 1.3, margin=3), # Spleen -- select_fn and margin are Task dependant
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
        #     RestoreLabelCustomd(keys='pred', ref_image='image', mode='nearest'),
        #     AsChannelLastd(keys='pred')
        # ]

        # return [
        #     ToTensord(keys=('image')),
        #     Activationsd(keys='image', sigmoid=True),
        #     AsDiscreted(keys='image', threshold_values=True, logit_thresh=0.51),
        #     RestoreLabelCustomd(keys='image', ref_image='image', mode='nearest'),
        #     AsChannelLastd(keys='image'),
        #     ToNumpyd(keys='image'),
        # ]

        # # Working for spleen so so
        return [
            ToTensord(keys=('image')),
            Activationsd(keys='image', sigmoid=True),
            AsDiscreted(keys='image', threshold_values=True, logit_thresh=0.51),
            SqueezeDimd(keys='image', dim=0),
            ToNumpyd(keys='image'),
        ]