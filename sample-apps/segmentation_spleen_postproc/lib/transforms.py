from copy import deepcopy

import numpy as np
import torch
from monai.networks.blocks import CRF
from monai.transforms.compose import Transform

# You can write your transforms here... which can be used in your train/infer tasks
from monailabel.utils.others.writer import Writer

from .utils import BIFSegUnary, maxflow2d, maxflow3d


class InteractiveSegmentationTransform(Transform):
    def _fetch_data(self, data, key):
        if key not in data.keys():
            raise ValueError("Key {} not found, present keys {}".format(key, data.keys()))

        return data[key]

    def _unfold_prob(self, data, axis=0):
        # in some cases, esp in MONAI Label, only background prob may be provided for binary classification
        # MONAI CRF requires these to be reconstructed into corresponding background and foreground prob
        # # unfold a single prob for background into bg/fg prob
        if data.shape[axis] == 1:
            data = np.concatenate([data, 1 - data], axis=axis)

        return data

    def _normalise_logits(self, data, axis=0):
        # check if logits is a true prob, if not then apply softmax
        if not np.allclose(np.mean(np.sum(data, axis=axis)), 1.0):
            print("found non normalized logits, normalizing using Softmax")
            data = torch.softmax(torch.from_numpy(data), dim=axis).numpy()

        return data


########################
#  Make Unary Transforms
########################
class MakeBIFSegUnaryd(InteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        logits: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        unary: str = "unary",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        scale_infty: float = 1.0,
        use_simplecrf=False,
    ) -> None:
        super(MakeBIFSegUnaryd, self).__init__()
        self.image = image
        self.logits = logits
        self.scribbles = scribbles
        self.meta_key_postfix = meta_key_postfix
        self.unary = unary
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.scale_infty = scale_infty
        self.use_simplecrf = use_simplecrf

    def __call__(self, data):
        d = dict(data)

        # copy meta data from pairwise input
        src_key = "_".join([self.image, self.meta_key_postfix])
        dst_key = "_".join([self.unary, self.meta_key_postfix])
        if src_key in d.keys():
            d[dst_key] = deepcopy(d[src_key])

        # create empty container for postprocessed label
        dst_shape = list(d[self.scribbles].shape)
        dst_shape[0] = 2
        d[self.unary] = np.zeros(dst_shape, dtype=np.float32)

        logits = self._fetch_data(d, self.logits)
        scribbles = self._fetch_data(d, self.scribbles)

        # check if input logits are compatible with BIFSeg opt
        if logits.shape[0] > 2:
            raise ValueError(
                "BIFSeg can only be applied to binary probabilities for now, received {}".format(logits.shape[0])
            )

        # attempt to unfold probability term from one dimension
        # assuming the one dimension given is background probability
        # we can get the foreground as 1-background and concat the two
        logits = self._unfold_prob(logits, axis=0)

        # make BIFSeg Unaries following Equation 7 from:
        # https://arxiv.org/pdf/1710.04043.pdf
        unary_term = BIFSegUnary(
            logits=logits,
            scribbles=scribbles,
            scribbles_bg_label=self.scribbles_bg_label,
            scribbles_fg_label=self.scribbles_fg_label,
            scale_infty=self.scale_infty,
            use_simplecrf=self.use_simplecrf,
        )
        d[self.unary] = unary_term

        return d


########################
########################

#######################
#  Optimiser Transforms
#######################
class ApplyCRFOptimisationd(InteractiveSegmentationTransform):
    def __init__(
        self,
        unary: str,
        pairwise: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "pred",
        iterations: int = 5,
        bilateral_weight: float = 5.0,
        gaussian_weight: float = 3.0,
        bilateral_spatial_sigma: float = 1.0,
        bilateral_color_sigma: float = 5.0,
        gaussian_spatial_sigma: float = 0.5,
        update_factor: float = 5.0,
        compatibility_kernel_range: int = 1,
        device: str = "cuda" if torch.cuda.is_available else "cpu",
    ) -> None:
        super(ApplyCRFOptimisationd, self).__init__()
        self.unary = unary
        self.pairwise = pairwise
        self.meta_key_postfix = meta_key_postfix
        self.post_proc_label = post_proc_label
        self.bilateral_weight = bilateral_weight
        self.gaussian_weight = gaussian_weight
        self.bilateral_spatial_sigma = bilateral_spatial_sigma
        self.bilateral_color_sigma = bilateral_color_sigma
        self.gaussian_spatial_sigma = gaussian_spatial_sigma
        self.update_factor = update_factor
        self.compatibility_kernel_range = compatibility_kernel_range
        self.iterations = iterations
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # copy meta data from pairwise input
        src_key = "_".join([self.pairwise, self.meta_key_postfix])
        dst_key = "_".join([self.post_proc_label, self.meta_key_postfix])
        if src_key in d.keys():
            d[dst_key] = deepcopy(d[src_key])

        # create empty container for postprocessed label
        d[self.post_proc_label] = np.zeros_like(d[self.pairwise])

        # read relevant terms from data
        unary_term = self._fetch_data(d, self.unary)
        pairwise_term = self._fetch_data(d, self.pairwise)

        # initialise MONAI's CRF layer
        crf_layer = CRF(
            bilateral_weight=self.bilateral_weight,
            gaussian_weight=self.gaussian_weight,
            bilateral_spatial_sigma=self.bilateral_spatial_sigma,
            bilateral_color_sigma=self.bilateral_color_sigma,
            gaussian_spatial_sigma=self.gaussian_spatial_sigma,
            update_factor=self.update_factor,
            compatibility_kernel_range=self.compatibility_kernel_range,
            iterations=self.iterations,
        )

        # add batch dimension for MONAI's CRF so it is in format [B, ?, X, Y, [Z]]
        unary_term = np.expand_dims(unary_term, axis=0)
        pairwise_term = np.expand_dims(pairwise_term, axis=0)

        # numpy to torch
        unary_term = torch.from_numpy(unary_term.astype(np.float32)).to(self.device)
        pairwise_term = torch.from_numpy(pairwise_term.astype(np.float32)).to(self.device)

        # run MONAI's CRF without any gradients
        with torch.no_grad():
            d[self.post_proc_label] = (
                torch.argmax(crf_layer(unary_term, pairwise_term), dim=1, keepdim=True)
                .squeeze_(dim=0)
                .detach()
                .cpu()
                .numpy()
            )

        return d


class ApplyGraphCutOptimisationd(InteractiveSegmentationTransform):
    def __init__(
        self,
        unary: str,
        pairwise: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "pred",
        lamda: float = 8.0,
        sigma: float = 0.1,
    ) -> None:
        super(ApplyGraphCutOptimisationd, self).__init__()
        self.unary = unary
        self.pairwise = pairwise
        self.meta_key_postfix = meta_key_postfix
        self.post_proc_label = post_proc_label
        self.lamda = lamda
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)

        # copy meta data from pairwise input
        src_key = "_".join([self.pairwise, self.meta_key_postfix])
        dst_key = "_".join([self.post_proc_label, self.meta_key_postfix])
        if src_key in d.keys():
            d[dst_key] = deepcopy(d[src_key])

        # create empty container for postprocessed label
        d[self.post_proc_label] = np.zeros_like(d[self.pairwise])

        # read relevant terms from data
        unary_term = self._fetch_data(d, self.unary)
        pairwise_term = self._fetch_data(d, self.pairwise)

        # check if input unary is compatible with GraphCut opt
        if unary_term.shape[0] > 2:
            raise ValueError(
                "GraphCut can only be applied to binary probabilities, received {}".format(unary_term.shape[0])
            )

        # attempt to unfold unary probability terms from one dimension
        # assuming the one dimension given is background probability
        # we can get the foreground as 1-background and concat the two
        unary_term = self._unfold_prob(unary_term, axis=0)

        # prepare data for SimpleCRF's GraphCut
        unary_term = np.moveaxis(unary_term, source=0, destination=-1)
        pairwise_term = np.moveaxis(pairwise_term, source=0, destination=-1)

        # run GraphCut
        spatial_dims = pairwise_term.ndim - 1
        run_3d = spatial_dims == 3
        if run_3d:
            post_proc_label = maxflow3d(pairwise_term, unary_term, lamda=self.lamda, sigma=self.sigma)
        else:
            post_proc_label = maxflow2d(pairwise_term, unary_term, lamda=self.lamda, sigma=self.sigma)

        post_proc_label = np.expand_dims(post_proc_label, axis=0)
        d[self.post_proc_label] = post_proc_label

        return d


#######################
#######################


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
