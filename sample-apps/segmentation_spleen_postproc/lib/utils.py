import maxflow
import numpy as np


def get_eps(data):
    return np.finfo(data.dtype).eps


def maxflow2d(image, prob, lamda=5, sigma=0.1):
    # lamda: weight of smoothing term
    # sigma: std of intensity values
    return maxflow.maxflow2d(image, prob, (lamda, sigma))


def maxflow3d(image, prob, lamda=5, sigma=0.1):
    # lamda: weight of smoothing term
    # sigma: std of intensity values
    return maxflow.maxflow3d(image, prob, (lamda, sigma))


def make_bifseg_unary(
    logits,
    scribbles,
    scribbles_bg_label=2,
    scribbles_fg_label=3,
    scale_infty=1,
    use_simplecrf=False,
):
    """
    Implements BIFSeg unary term from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    BIFSeg unary term is constructed using Equation 7 on page 4 of the above mentioned paper.
    """

    # fetch the data for probabilities and scribbles
    prob_shape = list(logits.shape)
    scrib_shape = list(scribbles.shape)

    # check if they have compatible shapes
    if prob_shape[1:] != scrib_shape[1:]:
        raise ValueError("shapes for logits and scribbles dont match")

    # expected input shape is [1, X, Y, [Z]], exit if first dimension doesnt satisfy
    if scrib_shape[0] != 1:
        raise ValueError("scribbles should have single channel first, received {}".format(scrib_shape[0]))

    # unfold a single logits for background into bg/fg logits (if needed)
    if prob_shape[0] == 1:
        logits = np.concatenate([logits, 1.0 - logits], axis=0)

    # for numerical stability, get rid of zeros
    # needed only for SimpleCRF, as internally it takes -log
    if use_simplecrf:
        logits += get_eps(logits)

    # extract background/foreground points from image
    if use_simplecrf:
        # swap fg with bg as -log taken inside simplecrf code
        background_pts = list(np.argwhere(scribbles == scribbles_fg_label))
        foreground_pts = list(np.argwhere(scribbles == scribbles_bg_label))
    else:
        # monai crf, use default
        background_pts = list(np.argwhere(scribbles == scribbles_bg_label))
        foreground_pts = list(np.argwhere(scribbles == scribbles_fg_label))

    # issue warning if no scribbles detected, the algorithm will still work
    # just need to inform user/researcher - in case it is unexpected
    if len(background_pts) == 0:
        print(
            "warining: no background scribbles received with label {}, available in scribbles {}".format(
                scribbles_bg_label, np.unique(scribbles)
            )
        )

    if len(foreground_pts) == 0:
        print(
            "warning: no foreground scribbles received with label {}, available in scribbles {}".format(
                scribbles_fg_label, np.unique(scribbles)
            )
        )

    # get infty and scale it
    infty = np.max(logits) * scale_infty

    # get predicted label y^
    y_hat = np.argmax(logits, axis=0)

    # copy probabilities
    unary_term = np.copy(logits)

    # get corrected labels from scribbles
    s_hat = [0] * len(background_pts) + [1] * len(foreground_pts)

    # update unary with Equation 7, including predicted label y^ and corrected labels s^
    eps = get_eps(unary_term)
    fg_bg_pts = background_pts + foreground_pts
    for s_h, fb_pt in zip(s_hat, fg_bg_pts):
        u_idx = tuple(fb_pt[1:])
        unary_term[(s_h,) + u_idx] = eps if y_hat[u_idx] == s_h else infty

    return unary_term
