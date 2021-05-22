import copy
import logging
from typing import Optional, Sequence, Union

import numpy as np
import skimage.measure as measure
import torch
from monai.config import KeysCollection
from monai.networks.blocks import CRF
from monai.transforms import Resize, generate_spatial_bounding_box, get_extreme_points
from monai.transforms.compose import MapTransform, Transform
from monai.transforms.spatial.dictionary import InterpolateModeSequence
from monai.utils import InterpolateMode, ensure_tuple_rep

logger = logging.getLogger(__name__)

# define epsilon for numerical stability
EPS = 7.0 / 3 - 4.0 / 3 - 1

# TODO:: Move to MONAI ??


class LargestCCd(MapTransform):
    def __init__(self, keys: KeysCollection, has_channel: bool = True):
        super().__init__(keys)
        self.has_channel = has_channel

    @staticmethod
    def get_largest_cc(label):
        largest_cc = np.zeros(shape=label.shape, dtype=label.dtype)
        for i, item in enumerate(label):
            item = measure.label(item, connectivity=1)
            if item.max() != 0:
                largest_cc[i, ...] = item == (np.argmax(np.bincount(item.flat)[1:]) + 1)
        return largest_cc

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = self.get_largest_cc(d[key] if self.has_channel else d[key][np.newaxis])
            d[key] = result if self.has_channel else result[0]
        return d


class ExtremePointsd(MapTransform):
    def __init__(self, keys: KeysCollection, result: str = "result", points: str = "points"):
        super().__init__(keys)
        self.result = result
        self.points = points

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            points = get_extreme_points(d[key])
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result][self.points] = np.array(points).astype(int).tolist()
        return d


class BoundingBoxd(MapTransform):
    def __init__(self, keys: KeysCollection, result: str = "result", bbox: str = "bbox"):
        super().__init__(keys)
        self.result = result
        self.bbox = bbox

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            bbox = generate_spatial_bounding_box(d[key])
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result][self.bbox] = np.array(bbox).astype(int).tolist()
        return d


class Restored(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        has_channel: bool = True,
        mode: InterpolateModeSequence = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        super().__init__(keys)
        self.ref_image = ref_image
        self.has_channel = has_channel
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        meta_dict = d[f"{self.ref_image}_{self.meta_key_postfix}"]
        for idx, key in enumerate(self.keys):
            result = d[key]
            current_size = result.shape[1:] if self.has_channel else result.shape
            spatial_shape = meta_dict["spatial_shape"]
            spatial_size = spatial_shape[-len(current_size) :]

            # Undo Spacing
            if np.any(np.not_equal(current_size, spatial_size)):
                resizer = Resize(spatial_size=spatial_size, mode=self.mode[idx])
                result = resizer(result, mode=self.mode[idx], align_corners=self.align_corners[idx])

            d[key] = result if len(result.shape) <= 3 else result[0]

            meta = d.get(f"{key}_{self.meta_key_postfix}")
            if meta is None:
                meta = dict()
                d[f"{key}_{self.meta_key_postfix}"] = meta
            meta["affine"] = meta_dict.get("original_affine")
        return d


class AddUnaryTermd(Transform):
    def __init__(
        self,
        ref_prob: str,
        unary: str = "unary",
        scribbles: str = "scribbles",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        channel_dim: int = 0,
        scale_infty: int = 1,
        use_simplecrf: bool = False,
    ):
        self.ref_prob = ref_prob
        self.unary = unary
        self.scribbles = scribbles
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.channel_dim = channel_dim
        self.scale_infty = scale_infty
        self.use_simplecrf = use_simplecrf

    def _apply(self, background_pts, foreground_pts, prob):
        # get infty
        infty = np.max(prob) * self.scale_infty

        # get label y^
        y_hat = np.argmax(prob, axis=self.channel_dim)

        unary_term = np.copy(prob)

        # override unary with 0 or infty following equation 7 from:
        # https://arxiv.org/pdf/1710.04043.pdf
        for bk_pt in background_pts:
            # unary_term[tuple(bk_pt)] = 0 if y_hat[(0,) + tuple(bk_pt[1:])] == 0 else infty
            # FIXME: find a more elegant way to determine index for unary term
            u_idx = (0,) + tuple(bk_pt) if self.channel_dim == 0 else tuple(bk_pt) + (0,)
            unary_term[u_idx] = EPS if y_hat[tuple(bk_pt)] == 0 else infty

        for fg_pt in foreground_pts:
            # unary_term[tuple(fg_pt)] = 0 if y_hat[(0,) + tuple(fg_pt[1:])] == 1 else infty
            # FIXME: find a more elegant way to determine index for unary term
            u_idx = (1,) + tuple(fg_pt) if self.channel_dim == 0 else tuple(fg_pt) + (1,)
            unary_term[u_idx] = EPS if y_hat[tuple(fg_pt)] == 1 else infty

        return unary_term

    def __call__(self, data):
        d = dict(data)

        if self.ref_prob in d.keys():
            prob = d[self.ref_prob]
        else:
            raise ValueError("Key {} not found, present keys {}".format(self.ref_prob, d.keys()))

        if self.scribbles in d.keys():
            scrib = d[self.scribbles]
        else:
            raise ValueError("Key {} not found, present keys {}".format(self.ref_prob, d.keys()))

        # fetch the data for probabilities and scribbles
        prob_shape = list(prob.shape)
        scrib_shape = list(scrib.shape)

        # check if they have compatible shapes
        if prob_shape != scrib_shape:
            raise ValueError("shapes for prob and scribbles dont match")

        # in some cases, esp in MONAI Label, only background prob may be provided for binary classification
        # MONAI CRF requires these to be reconstructed into corresponding background and foreground prob
        if prob_shape[self.channel_dim] == 1:  # unfold a single prob for background into bg/fg prob
            prob = np.concatenate([(prob).copy(), (1.0 - prob).copy()], axis=self.channel_dim)

        # for numerical stability, get rid of zeros
        # FIXME: find another way to do this only when needed
        prob += EPS

        scrib = np.squeeze(scrib)  # 4d to 3d drop first dim in [1, x, y, z]

        # extract background/foreground points from image
        background_pts = np.argwhere(scrib == self.scribbles_bg_label)
        foreground_pts = np.argwhere(scrib == self.scribbles_fg_label)

        if len(background_pts) == 0:
            logger.info(
                f"no background scribbles received with label {self.scribbles_bg_label}, available in scribbles {np.unique(scrib)}"
            )

        if len(foreground_pts) == 0:
            logger.info(
                f"no foreground scribbles received with label {self.scribbles_fg_label}, available in scribbles {np.unique(scrib)}"
            )

        if self.use_simplecrf:
            # swap fg with bg as -log taken inside simplecrf code
            d[self.unary] = self._apply(foreground_pts, background_pts, prob)
        else:  # monai crf
            d[self.unary] = self._apply(background_pts, foreground_pts, prob)
        return d


class ApplyMONAICRFPostProcd(Transform):
    def __init__(
        self,
        unary: str,
        pairwise: str,
        post_proc_label: str = "pred",
        iterations: int = 5,
        bilateral_weight: float = 3.0,
        gaussian_weight: float = 1.0,
        bilateral_spatial_sigma: float = 5.0,
        bilateral_color_sigma: float = 0.5,
        gaussian_spatial_sigma: float = 5.0,
        compatibility_kernel_range: int = 1,
        device: str = "cuda" if torch.cuda.is_available else "cpu",
    ):
        self.unary = unary
        self.pairwise = pairwise
        self.post_proc_label = post_proc_label
        self.device = device

        self.crf_layer = CRF(
            iterations,
            bilateral_weight,
            gaussian_weight,
            bilateral_spatial_sigma,
            bilateral_color_sigma,
            gaussian_spatial_sigma,
            compatibility_kernel_range,
        )

    def __call__(self, data):
        d = dict(data)

        if self.unary in d.keys():
            unary_term = d[self.unary].float()
        else:
            raise ValueError("Key {} not found, present keys {}".format(self.unary, d.keys()))

        if self.pairwise in d.keys():
            pairwise_term = d[self.pairwise].float()
        else:
            raise ValueError("Key {} not found, present keys {}".format(self.pairwise, d.keys()))

        unary_term = unary_term.to(self.device)
        pairwise_term = pairwise_term.to(self.device)

        d[self.post_proc_label] = torch.argmax(self.crf_layer(unary_term, pairwise_term), dim=1, keepdims=True)

        # copy meta data from pairwise input
        if self.pairwise + "_meta_dict" in d.keys():
            d[self.post_proc_label + "_meta_dict"] = copy.deepcopy(d[self.pairwise + "_meta_dict"])

        return d
