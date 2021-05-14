import numpy as np
import torch
import denseCRF3D # pip install simplecrf

import logging

from monai.networks.blocks import CRF
from monai.transforms import Transform

logger = logging.getLogger(__name__)

# define epsilon for numerical stability
# help from: https://stackoverflow.com/a/25155518
EPS = 7./3 - 4./3 - 1

# Maybe these can go in MONAI, not sure at the moment
class AddUnaryTermd(Transform):
    def __init__(
        self,
        ref_prob,
        unary: str = "unary",
        scribbles: str = "scribbles",
        sc_background_label: int = 2,
        sc_foreground_label: int = 3,
        channel_dim: int = 0,
        scale_infty: int = 1,
        use_simplecrf: bool = False,

    ):
        self.ref_prob = ref_prob
        self.unary = unary
        self.scribbles = scribbles
        self.sc_background_label = sc_background_label
        self.sc_foreground_label = sc_foreground_label
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
    
    def _apply2(self, background_pts, foreground_pts, prob):
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
            unary_term[u_idx] = infty if y_hat[tuple(bk_pt)] == 0 else EPS

        for fg_pt in foreground_pts:
            # unary_term[tuple(fg_pt)] = 0 if y_hat[(0,) + tuple(fg_pt[1:])] == 1 else infty
            # FIXME: find a more elegant way to determine index for unary term
            u_idx = (1,) + tuple(fg_pt) if self.channel_dim == 0 else tuple(fg_pt) + (1,)
            unary_term[u_idx] = infty if y_hat[tuple(fg_pt)] == 1 else EPS

        return unary_term

    def __call__(self, data):
        d = dict(data)
        prob = d[self.ref_prob]
        scrib = d[self.scribbles]

        prob_shape = list(prob.shape)
        scrib_shape = list(scrib.shape)

        if prob_shape != scrib_shape:
            raise ValueError('shapes for prob and scribbles dont match')

        if prob_shape[self.channel_dim] == 1: # unfold a single prob for background into bg/fg prob
            prob = np.concatenate([(prob).copy(), (1.-prob).copy()], axis=self.channel_dim)

        # for numerical stability, get rid of zeros
        # FIXME: find another way to do this only when needed
        prob += EPS 

        scrib = np.squeeze(scrib) # 4d to 3d drop first dim in [1, x, y, z]

        # extract background/foreground points from image
        # help from: https://stackoverflow.com/a/58087561
        background_pts = np.argwhere(scrib == self.sc_background_label)
        foreground_pts = np.argwhere(scrib ==  self.sc_foreground_label)

        if self.use_simplecrf:
            # swap fg with bg as -log taken inside simplecrf code
            d[self.unary] = self._apply(foreground_pts, background_pts, prob)
        else: # monai crf
            d[self.unary] = self._apply(background_pts, foreground_pts, prob)
        return d

class ApplyCRFPostProcd(Transform):
    def __init__(
        self,
        unary: str,
        pairwise: str,
        post_proc_label: str = 'postproc',
        iterations: int = 5, 
        bilateral_weight: float = 3.0,
        gaussian_weight: float = 1.0,
        bilateral_spatial_sigma: float = 5.0,
        bilateral_color_sigma: float = 0.5,
        gaussian_spatial_sigma: float = 5.0,
        compatibility_kernel_range: float = 1,
        device = torch.device('cpu'),
        use_simplecrf = True
    ):
        self.unary = unary
        self.pairwise = pairwise
        self.post_proc_label = post_proc_label
        self.use_simplecrf = use_simplecrf
        if self.use_simplecrf:
            self.simplecrf_params = {}
            self.simplecrf_params['MaxIterations'] = iterations
            
            # Gaussian
            self.simplecrf_params['PosW'] = gaussian_weight 
            self.simplecrf_params['PosRStd'] = int(gaussian_spatial_sigma) # row
            self.simplecrf_params['PosCStd'] = int(gaussian_spatial_sigma) # col
            self.simplecrf_params['PosZStd'] = int(gaussian_spatial_sigma) # depth (z direction)

            # Bilateral spatial
            bilateral_weight = 5
            self.simplecrf_params['BilateralW'] = bilateral_weight
            self.simplecrf_params['BilateralRStd'] = bilateral_spatial_sigma # row
            self.simplecrf_params['BilateralCStd'] = bilateral_spatial_sigma # col
            self.simplecrf_params['BilateralZStd'] = bilateral_spatial_sigma # depth (z direction)
            self.simplecrf_params['ModalityNum'] = 1

            # Bilateral color
            bilateral_color_sigma = 5
            self.simplecrf_params['BilateralModsStds'] = (bilateral_color_sigma,)
        else:
            self.device = device

            self.crf_layer = CRF(
                    iterations, 
                    bilateral_weight,
                    gaussian_weight,
                    bilateral_spatial_sigma,
                    bilateral_color_sigma,
                    gaussian_spatial_sigma,
                    compatibility_kernel_range
                    )

    def __call__(self, data):
        d = dict(data)
        unary_term = d[self.unary].float()
        pairwise_term = d[self.pairwise].float()

        if self.use_simplecrf:
            # SimpleCRF expects channel last, e.g. [144, 144, 144, 2] or [144, 144, 144, 1]
            # however input to this transform is channel first with batch dimensions
            # i.e. [1, 2, 144, 144, 144] or [1, 1, 144, 144, 144]
            # so we will convert it and also convert torch.Tensor to np array where applicable

            if isinstance(unary_term, torch.Tensor):
                unary_term = unary_term.cpu().numpy()
            
            if isinstance(pairwise_term, torch.Tensor):
                pairwise_term = pairwise_term.cpu().numpy()

            unary_term = np.transpose(np.squeeze(unary_term, axis=0), (1, 2, 3, 0)) # channel last + squeeze
            pairwise_term = np.transpose(np.squeeze(pairwise_term, axis=0), (1, 2, 3, 0)) # channel last + squeeze
        
            min_p = pairwise_term.min()
            max_p = pairwise_term.max()
            pairwise_term = (((pairwise_term-min_p)/(max_p-min_p)) * 255).astype(np.uint8)

            out = denseCRF3D.densecrf3d(pairwise_term, unary_term, self.simplecrf_params).astype(np.float64)
            out = np.expand_dims(out, axis=0)
            out = np.expand_dims(out, axis=0)

            d[self.post_proc_label] = out
        else:
            unary_term = d[self.unary].to(self.device)
            pairwise_term = d[self.pairwise].to(self.device)
            
            # # Experimental, quantize pairwise term to N bits
            # min_p = pairwise_term.min()
            # max_p = pairwise_term.max()

            # pairwise_term = (pairwise_term-min_p)/(max_p - min_p) # in range (0, 1)
            # bits = 10
            # pairwise_term = (2**bits) * torch.round(pairwise_term * (2**bits))

            d[self.post_proc_label] = torch.argmax(self.crf_layer(unary_term, pairwise_term), dim=1, keepdims=True)
        return d