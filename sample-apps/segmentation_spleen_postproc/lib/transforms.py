import copy
import logging

import numpy as np
import torch
from monai.networks.blocks import CRF
from monai.transforms.compose import Transform

# You can write your transforms here... which can be used in your train/infer tasks
from monailabel.utils.others.writer import Writer

logger = logging.getLogger(__name__)

# define epsilon for numerical stability
EPS = 7.0 / 3 - 4.0 / 3 - 1


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
            # FIXME: find a more elegant way to determine index for unary term
            u_idx = (0,) + tuple(bk_pt) if self.channel_dim == 0 else tuple(bk_pt) + (0,)
            unary_term[u_idx] = EPS if y_hat[tuple(bk_pt)] == 0 else infty

        for fg_pt in foreground_pts:
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


class ApplyCRFPostProcd(Transform):
    def __init__(
        self,
        unary: str,
        pairwise: str,
        post_proc_label: str = "pred",
        bilateral_weight: float = 5.0,
        gaussian_weight: float = 3.0,
        bilateral_spatial_sigma: float = 1.0,
        bilateral_color_sigma: float = 5.0,
        gaussian_spatial_sigma: float = 0.5,
        update_factor: float = 5.0,
        compatibility_kernel_range: int = 1,
        iterations: int = 5,
        device: str = "cuda" if torch.cuda.is_available else "cpu",
    ):
        self.unary = unary
        self.pairwise = pairwise
        self.post_proc_label = post_proc_label
        self.device = device

        self.crf_layer = CRF(
            bilateral_weight=bilateral_weight,
            gaussian_weight=gaussian_weight,
            bilateral_spatial_sigma=bilateral_spatial_sigma,
            bilateral_color_sigma=bilateral_color_sigma,
            gaussian_spatial_sigma=gaussian_spatial_sigma,
            update_factor=update_factor,
            compatibility_kernel_range=compatibility_kernel_range,
            iterations=iterations,
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


class WriteLogits(Transform):
    def __init__(self, key, result="result"):
        self.key = key
        self.result = result

    def __call__(self, data):
        d = dict(data)
        writer = Writer(label=self.key)

        file, _ = writer(d)
        if data.get(self.result) is None:
            d[self.result] = {}
        d[self.result][self.key] = file
        return d
