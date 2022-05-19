# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import numpy as np
import numpymaxflow
import torch
from monai.networks.layers import GaussianMixtureModel

logger = logging.getLogger(__name__)


def get_eps(data):
    return np.finfo(data.dtype).eps


def maxflow(image, prob, lamda=5, sigma=0.1):
    # lamda: weight of smoothing term
    # sigma: std of intensity values
    return numpymaxflow.maxflow(image, prob, lamda, sigma)


def make_iseg_unary(
    prob,
    scribbles,
    scribbles_bg_label=2,
    scribbles_fg_label=3,
):
    """
    Implements ISeg unary term from the following paper:
    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)
    ISeg unary term is constructed using Equation 7 on page 4 of the above mentioned paper.
    """

    # fetch the data for probabilities and scribbles
    prob_shape = list(prob.shape)
    scrib_shape = list(scribbles.shape)

    # check if they have compatible shapes
    if prob_shape[1:] != scrib_shape[1:]:
        raise ValueError("shapes for prob and scribbles dont match")

    # expected input shape is [1, X, Y, [Z]], exit if first dimension doesnt comply
    if scrib_shape[0] != 1:
        raise ValueError(f"scribbles should have single channel first, received {scrib_shape[0]}")

    # unfold a single prob for background into bg/fg prob (if needed)
    if prob_shape[0] == 1:
        prob = np.concatenate([prob, 1.0 - prob], axis=0)

    mask = np.concatenate([scribbles == scribbles_bg_label, scribbles == scribbles_fg_label], axis=0)

    # issue a warning if no scribbles detected, the algorithm will still work
    # just need to inform user/researcher - in case it is unexpected
    if not np.any(mask[0, ...]):
        logging.info(
            "warning: no background scribbles received with label {}, available in scribbles {}".format(
                scribbles_bg_label, np.unique(scribbles)
            )
        )

    if not np.any(mask[1, ...]):
        logging.info(
            "warning: no foreground scribbles received with label {}, available in scribbles {}".format(
                scribbles_fg_label, np.unique(scribbles)
            )
        )

    # copy probabilities
    unary_term = np.copy(prob)

    # for numerical stability, get rid of zeros
    eps = get_eps(unary_term)

    equal_term = 1.0 - eps
    no_equal_term = eps

    # update unary with Equation 7
    unary_term[mask] = equal_term
    mask = np.flip(mask, axis=0)
    unary_term[mask] = no_equal_term

    return unary_term


def make_histograms(image, scrib, scribbles_bg_label, scribbles_fg_label, alpha_bg=1, alpha_fg=1, bins=32):
    # alpha forms the pseudo-counts for Dirichlet distribution used here as
    # conjugate prior to histogram distributions which enables us to make
    # histograms work in cases where only foreground or only background scribbles are provide

    # alpha can be:
    # - a scalar, where it is expanded into a list of size==bins
    # - a list of scalars, where it is checked against size==bins and applied

    def expand_pseudocounts(alpha):
        # expand pseudo-counts into array if needed
        if not isinstance(alpha, list):
            alpha = [alpha] * bins
        elif len(alpha) != bins:
            raise ValueError(
                "pseudo-counts size does not match number of bins in histogram, received: {} | num_bins {}".format(
                    len(alpha), bins
                )
            )
        alpha = np.array(alpha)
        return alpha

    alpha_bg = expand_pseudocounts(alpha_bg)
    alpha_fg = expand_pseudocounts(alpha_fg)

    # collect background voxels
    values = image[scrib == scribbles_bg_label]
    # generate histogram for background
    bg_hist, _ = np.histogram(values, bins=bins, range=(0, 1), density=False)

    # collect foreground voxels
    values = image[scrib == scribbles_fg_label]
    # generate histrogram for foreground
    fg_hist, fg_bin_edges = np.histogram(values, bins=bins, range=(0, 1), density=False)

    # add Dirichlet distribution as conjugate prior for our histogram distributions
    bg_hist = bg_hist + alpha_bg
    fg_hist = fg_hist + alpha_fg

    # normalise histograms
    bg_hist = bg_hist / np.sum(bg_hist)
    fg_hist = fg_hist / np.sum(fg_hist)

    # normalise histograms and return
    return bg_hist.astype(np.float32), fg_hist.astype(np.float32), fg_bin_edges.astype(np.float32)


def make_likelihood_image_histogram(
    image, scrib, scribbles_bg_label, scribbles_fg_label, num_bins=64, return_label=False
):
    # normalise image in range [0, 1] if needed
    min_img = np.min(image)
    max_img = np.max(image)
    if min_img < 0.0 or max_img > 1.0:
        image = (image - min_img) / (max_img - min_img)

    # generate histograms for background/foreground
    bg_hist, fg_hist, bin_edges = make_histograms(
        image, scrib, scribbles_bg_label, scribbles_fg_label, alpha_bg=1, alpha_fg=1, bins=num_bins
    )

    # lookup values for each voxel for generating background/foreground probabilities
    dimage = np.digitize(image, bin_edges[:-1]) - 1
    fprob = fg_hist[dimage]
    bprob = bg_hist[dimage]
    retprob = np.concatenate([bprob, fprob], axis=0)

    # if needed, convert to discrete labels instead of probability
    if return_label:
        retprob = np.expand_dims(np.argmax(retprob, axis=0), axis=0).astype(np.float32)

    return retprob


def learn_and_apply_gmm_monai(image, scrib, scribbles_bg_label, scribbles_fg_label, num_mixtures):
    # this function is limited to binary segmentation at the moment
    n_classes = 2

    # make trimap
    trimap = np.zeros_like(scrib).astype(np.int32)

    # fetch anything that is not scribbles
    not_scribbles = ~((scrib == scribbles_bg_label) | (scrib == scribbles_fg_label))

    # set these to -1 == unused
    trimap[not_scribbles] = -1

    # set background scrib to 0
    trimap[scrib == scribbles_bg_label] = 0
    # set foreground scrib to 1
    trimap[scrib == scribbles_fg_label] = 1

    # add empty channel to image and scrib to be inline with pytorch layout
    image = np.expand_dims(image, axis=0)
    trimap = np.expand_dims(trimap, axis=0)

    # transfer everything to pytorch tensor
    # we use CUDA as GMM from MONAI is only available on CUDA atm (29/04/2022)
    # if no cuda device found, then exit now
    if not torch.cuda.is_available():
        raise OSError("Unable to find CUDA device, check your torch/monai installation")

    device = "cuda"
    image = torch.from_numpy(image).type(torch.float32).to(device)
    trimap = torch.from_numpy(trimap).type(torch.int32).to(device)

    # initialise our GMM
    gmm = GaussianMixtureModel(
        image.size(1),
        mixture_count=n_classes,
        mixture_size=num_mixtures,
        verbose_build=False,
    )

    # learn gmm from image and trimap
    gmm.learn(image, trimap)

    # apply gmm on image
    gmm_output = gmm.apply(image)

    # return output
    return gmm_output.squeeze(0).cpu().numpy()


def make_likelihood_image_gmm(
    image,
    scrib,
    scribbles_bg_label,
    scribbles_fg_label,
    num_mixtures=20,
    return_label=False,
):
    # learn gmm and apply to image, return output label prob
    retprob = learn_and_apply_gmm_monai(
        image=image,
        scrib=scrib,
        scribbles_bg_label=scribbles_bg_label,
        scribbles_fg_label=scribbles_fg_label,
        num_mixtures=num_mixtures,
    )

    # if needed, convert to discrete labels instead of probability
    if return_label:
        retprob = np.expand_dims(np.argmax(retprob, axis=0), axis=0).astype(np.float32)

    return retprob
