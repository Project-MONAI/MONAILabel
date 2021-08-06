from copy import deepcopy
from typing import Optional

import denseCRF
import denseCRF3D
import numpy as np
import torch
from monai.networks.blocks import CRF
from monai.transforms.compose import Transform

from monailabel.utils.others.writer import Writer

from .utils import interactive_maxflow2d, interactive_maxflow3d, make_bifseg_unary, maxflow2d, maxflow3d


#####################################
# Interactive Segmentation Transforms
#
# Base class for implementing common
# functionality for int. seg. tx
#####################################
class InteractiveSegmentationTransform(Transform):
    def _fetch_data(self, data, key):
        if key not in data.keys():
            raise ValueError("Key {} not found, present keys {}".format(key, data.keys()))

        return data[key]

    def _unfold_prob(self, prob, axis=0):
        # in some cases, esp in MONAILabel, only background prob may be present
        # optimisers (e.g. CRF, GraphCut) require these to be reconstructed in two bg/fg prob
        # assuming prob is background, form foreground as 1-background and concat the two
        if prob.shape[axis] == 1:
            prob = np.concatenate([prob, 1 - prob], axis=axis)

        return prob

    def _normalise_logits(self, data, axis=0):
        # check if logits is a true prob, if not then apply softmax
        if not np.allclose(np.sum(data, axis=axis), 1.0):
            print("found non normalized logits, normalizing using Softmax")
            data = torch.softmax(torch.from_numpy(data), dim=axis).numpy()

        return data

    def _copy_affine(self, d, src, dst):
        # make keys
        src_key = "_".join([src, self.meta_key_postfix])
        dst_key = "_".join([dst, self.meta_key_postfix])

        # check if keys exists, if so then copy affine info
        if src_key in d.keys() and "affine" in d[src_key]:
            # create a new destination dictionary if needed
            d[dst_key] = {} if dst_key not in d.keys() else d[dst_key]

            # copy over affine information
            d[dst_key]["affine"] = deepcopy(d[src_key]["affine"])

        return d


#####################################
#####################################

########################
#  Make Unary Transforms
########################
class MakeBIFSegUnaryd(InteractiveSegmentationTransform):
    """
    Implements forming BIFSeg unary term from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    BIFSeg unary term is constructed using Equation 7 on page 4 of the above mentioned paper.
    This unary term along with a pairwise term (e.g. input image volume) form Equation 5 in the paper,
    which defines an energy to be minimised. Equation 5 can be optimised using an appropriate
    optimisation method (e.g. CRF, GraphCut etc), which is implemented here as an additional transform.

    Usage Example::

    Compose(
        [
            # unary term maker
            MakeBIFSegUnaryd(
                image="image",
                logits="logits",
                scribbles="label",
                unary="unary",
                scribbles_bg_label=2,
                scribbles_fg_label=3,
                scale_infty=1e6,
            ),
            # optimiser
            ApplyCRFOptimisationd(unary="unary", pairwise="image", post_proc_label="pred"),
        ]
    )
    """

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

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from image input
        self._copy_affine(d, self.image, self.unary)

        # read relevant terms from data
        logits = self._fetch_data(d, self.logits)
        scribbles = self._fetch_data(d, self.scribbles)

        # check if input logits are compatible with BIFSeg opt
        if logits.shape[0] > 2:
            raise ValueError(
                "BIFSeg can only be applied to binary probabilities for now, received {}".format(logits.shape[0])
            )

        # attempt to unfold probability term
        logits = self._unfold_prob(logits, axis=0)

        # make BIFSeg Unaries following Equation 7 from:
        # https://arxiv.org/pdf/1710.04043.pdf
        unary_term = make_bifseg_unary(
            logits=logits,
            scribbles=scribbles,
            scribbles_bg_label=self.scribbles_bg_label,
            scribbles_fg_label=self.scribbles_fg_label,
            scale_infty=self.scale_infty,
        )
        d[self.unary] = unary_term

        return d


########################
########################

#################################################
# Hybrid Transforms
# (both MakeUnary+Optimiser in single method)
# uses SimpleCRF's interactive_maxflowNd() method
#################################################
class ApplyBIFSegGraphCutPostProcd(InteractiveSegmentationTransform):
    """
    Transform wrapper around SimpleCRF's Interactive GraphCut MaxFlow implementation for BIFSeg.

    This is a hybrid transform, which covers both Make*Unaryd + Optimiser and hence does not
    need an additional optimiser.

    BIFSeg unaries are made and resulting energy function optimised using GraphCut inside
    SimpleCRF's interactive segmentation function.

    Usage Example::

    Compose(
        [
            ApplyBIFSegGraphCutPostProcd(
                image="image",
                logits="logits",
                scribbles="label",
                post_proc_label="pred",
                scribbles_bg_label=2,
                scribbles_fg_label=3,
                lamda=10.0,
                sigma=15.0,
            ),
        ]
    )
    """

    def __init__(
        self,
        image: str,
        logits: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "pred",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        lamda: float = 8.0,
        sigma: float = 0.1,
    ) -> None:
        super(ApplyBIFSegGraphCutPostProcd, self).__init__()
        self.image = image
        self.logits = logits
        self.scribbles = scribbles
        self.meta_key_postfix = meta_key_postfix
        self.post_proc_label = post_proc_label
        self.lamda = lamda
        self.sigma = sigma
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.post_proc_label)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        logits = self._fetch_data(d, self.logits)
        scribbles = self._fetch_data(d, self.scribbles)

        # start forming user interaction input
        scribbles = np.concatenate(
            [scribbles == self.scribbles_bg_label, scribbles == self.scribbles_fg_label], axis=0
        ).astype(np.uint8)

        # check if input logit is compatible with GraphCut opt
        if logits.shape[0] > 2:
            raise ValueError(
                "GraphCut can only be applied to binary probabilities, received {}".format(logits.shape[0])
            )

        # attempt to unfold probability term
        logits = self._unfold_prob(logits, axis=0)

        # prepare data for SimpleCRF's Interactive GraphCut (BIFSeg)
        image = np.moveaxis(image, source=0, destination=-1)
        logits = np.moveaxis(logits, source=0, destination=-1)
        scribbles = np.moveaxis(scribbles, source=0, destination=-1)

        # run GraphCut
        spatial_dims = image.ndim - 1
        run_3d = spatial_dims == 3
        if run_3d:
            post_proc_label = interactive_maxflow3d(image, logits, scribbles, lamda=self.lamda, sigma=self.sigma)
        else:
            # 2D is not yet tested within this framework
            post_proc_label = interactive_maxflow2d(image, logits, scribbles, lamda=self.lamda, sigma=self.sigma)

        post_proc_label = np.expand_dims(post_proc_label, axis=0)
        d[self.post_proc_label] = post_proc_label

        return d


#################################################
#################################################

#######################
#  Optimiser Transforms
#######################
class ApplyCRFOptimisationd(InteractiveSegmentationTransform):
    """
    Generic MONAI CRF optimisation transform.

    This can be used in conjuction with any Make*Unaryd transform
    (e.g. MakeBIFSegUnaryd from above for implementing BIFSeg unary term).
    It optimises a typical energy function for interactive segmentation methods using MONAI's CRF layer,
    e.g. Equation 5 from https://arxiv.org/pdf/1710.04043.pdf.

    Usage Example::

    Compose(
        [
            # unary term maker
            MakeBIFSegUnaryd(
                image="image",
                logits="logits",
                scribbles="label",
                unary="unary",
                scribbles_bg_label=2,
                scribbles_fg_label=3,
                scale_infty=1e6,
            ),
            # optimiser
            ApplyCRFOptimisationd(unary="unary", pairwise="image", post_proc_label="pred"),
        ]
    )
    """

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
        compatibility_matrix: Optional[torch.Tensor] = None,
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
        self.compatibility_matrix = compatibility_matrix
        self.iterations = iterations
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from pairwise input
        self._copy_affine(d, self.pairwise, self.post_proc_label)

        # read relevant terms from data
        unary_term = self._fetch_data(d, self.unary)
        pairwise_term = self._fetch_data(d, self.pairwise)

        # initialise MONAI's CRF layer
        crf_layer = CRF(
            iterations=self.iterations,
            bilateral_weight=self.bilateral_weight,
            gaussian_weight=self.gaussian_weight,
            bilateral_spatial_sigma=self.bilateral_spatial_sigma,
            bilateral_color_sigma=self.bilateral_color_sigma,
            gaussian_spatial_sigma=self.gaussian_spatial_sigma,
            update_factor=self.update_factor,
            compatibility_matrix=self.compatibility_matrix,
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


class ApplySimpleCRFOptimisationd(InteractiveSegmentationTransform):
    """
    Generic SimpleCRF's CRF optimisation transform.

    This can be used in conjuction with any Make*Unaryd transform
    (e.g. MakeBIFSegUnaryd from above for implementing BIFSeg unary term).
    It optimises a typical energy function for interactive segmentation methods using SimpleCRF's CRF method,
    e.g. Equation 5 from https://arxiv.org/pdf/1710.04043.pdf.

    Usage Example::

    Compose(
        [
            # unary term maker
            MakeBIFSegUnaryd(
                image="image",
                logits="logits",
                scribbles="label",
                unary="unary",
                scribbles_bg_label=2,
                scribbles_fg_label=3,
                scale_infty=1e6,
            ),
            # optimiser
            ApplySimpleCRFOptimisationd(
                unary="unary",
                pairwise="image",
                post_proc_label="pred",
            ),
        ]
    )
    """

    def __init__(
        self,
        unary: str,
        pairwise: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "pred",
        iterations: int = 5,
        bilateral_weight: int = 5,
        gaussian_weight: int = 3,
        bilateral_spatial_sigma: int = 1,
        bilateral_color_sigma: int = 5,
        gaussian_spatial_sigma: int = 1,
        number_of_modalities: int = 1,
    ) -> None:
        super(ApplySimpleCRFOptimisationd, self).__init__()
        self.unary = unary
        self.pairwise = pairwise
        self.meta_key_postfix = meta_key_postfix
        self.post_proc_label = post_proc_label
        self.iterations = iterations
        self.bilateral_weight = bilateral_weight
        self.gaussian_weight = gaussian_weight
        self.bilateral_spatial_sigma = bilateral_spatial_sigma
        self.bilateral_color_sigma = bilateral_color_sigma
        self.gaussian_spatial_sigma = gaussian_spatial_sigma
        self.number_of_modalities = number_of_modalities

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from pairwise input
        d = self._copy_affine(d, src=self.pairwise, dst=self.post_proc_label)

        # read relevant terms from data
        unary_term = self._fetch_data(d, self.unary)
        pairwise_term = self._fetch_data(d, self.pairwise)

        # SimpleCRF expects uint8 for pairwise_term
        # scaling of pairwise_term handled in pre_transforms should be in range [0, 1]
        # just in case it is needed, leaving a basic scaling method here
        # min_p = pairwise_term.min()
        # max_p = pairwise_term.max()
        # pairwise_term = (((pairwise_term - min_p) / (max_p - min_p)) * 255).astype(np.uint8)
        pairwise_term = (pairwise_term * 255).astype(np.uint8)

        # prepare data for SimpleCRF's CRF
        unary_term = np.moveaxis(unary_term, source=0, destination=-1)
        pairwise_term = np.moveaxis(pairwise_term, source=0, destination=-1)

        # run SimpleCRF's CRF
        spatial_dims = pairwise_term.ndim - 1
        run_3d = spatial_dims == 3
        if run_3d:
            simplecrf_params = {}
            simplecrf_params["MaxIterations"] = self.iterations

            # Gaussian
            simplecrf_params["PosW"] = self.gaussian_weight
            simplecrf_params["PosRStd"] = self.gaussian_spatial_sigma  # row
            simplecrf_params["PosCStd"] = self.gaussian_spatial_sigma  # col
            simplecrf_params["PosZStd"] = self.gaussian_spatial_sigma  # depth (z direction)

            # Bilateral spatial
            simplecrf_params["BilateralW"] = self.bilateral_weight
            simplecrf_params["BilateralRStd"] = self.bilateral_spatial_sigma  # row
            simplecrf_params["BilateralCStd"] = self.bilateral_spatial_sigma  # col
            simplecrf_params["BilateralZStd"] = self.bilateral_spatial_sigma  # depth (z direction)
            simplecrf_params["ModalityNum"] = self.number_of_modalities

            # Bilateral color
            simplecrf_params["BilateralModsStds"] = (self.bilateral_color_sigma,)

            post_proc_label = denseCRF3D.densecrf3d(pairwise_term, unary_term, simplecrf_params)
        else:
            # 2D is not yet tested within this framework
            # 2D parameters are different, so prepare them
            simplecrf_params = (
                self.bilateral_weight,
                self.bilateral_spatial_sigma,
                self.bilateral_color_sigma,
                self.gaussian_weight,
                self.gaussian_spatial_sigma,
                self.iterations,
            )

            post_proc_label = denseCRF.densecrf(pairwise_term, unary_term, simplecrf_params)

        post_proc_label = np.expand_dims(post_proc_label, axis=0)
        d[self.post_proc_label] = post_proc_label

        return d


class ApplyGraphCutOptimisationd(InteractiveSegmentationTransform):
    """
    Generic GraphCut optimisation transform.

    This can be used in conjuction with any Make*Unaryd transform
    (e.g. MakeBIFSegUnaryd from above for implementing BIFSeg unary term).
    It optimises a typical energy function for interactive segmentation methods using SimpleCRF's GraphCut method,
    e.g. Equation 5 from https://arxiv.org/pdf/1710.04043.pdf.

    Usage Example::

    Compose(
        [
            # unary term maker
            MakeBIFSegUnaryd(
                image="image",
                logits="logits",
                scribbles="label",
                unary="unary",
                scribbles_bg_label=2,
                scribbles_fg_label=3,
                scale_infty=1e6,
            ),
            # optimiser
            ApplyGraphCutOptimisationd(
                unary="unary",
                pairwise="image",
                post_proc_label="pred",
                lamda=10.0,
                sigma=15.0,
            ),
        ]
    )
    """

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

        # copy affine meta data from pairwise input
        self._copy_affine(d, self.pairwise, self.post_proc_label)

        # read relevant terms from data
        unary_term = self._fetch_data(d, self.unary)
        pairwise_term = self._fetch_data(d, self.pairwise)

        # check if input unary is compatible with GraphCut opt
        if unary_term.shape[0] > 2:
            raise ValueError(
                "GraphCut can only be applied to binary probabilities, received {}".format(unary_term.shape[0])
            )

        # attempt to unfold probability term
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
            # 2D is not yet tested within this framework
            post_proc_label = maxflow2d(pairwise_term, unary_term, lamda=self.lamda, sigma=self.sigma)

        post_proc_label = np.expand_dims(post_proc_label, axis=0)
        d[self.post_proc_label] = post_proc_label

        return d


#######################
#######################


########################
#  Logits Save Transform
########################
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


########################
########################
