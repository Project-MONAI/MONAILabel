import numpy as np
from numpy.lib.arraysetops import isin
import torch
import denseCRF3D

import logging

from monai.networks.blocks import CRF
from monai.transforms import Transform

logger = logging.getLogger(__name__)

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

    ):
        self.ref_prob = ref_prob
        self.unary = unary
        self.scribbles = scribbles
        self.sc_background_label = sc_background_label
        self.sc_foreground_label = sc_foreground_label
        self.channel_dim = channel_dim
        self.scale_infty = scale_infty

    def _apply(self, background_pts, foreground_pts, prob):
        # get infty
        infty = np.max(prob) * self.scale_infty

        # get label y^
        # FIXME: find an elegant way of "keeping_dims" for argmax
        y_hat = np.expand_dims(np.argmax(prob, axis=self.channel_dim), axis=self.channel_dim)
        
        unary_term = np.copy(prob)

        # override unary with 0 or infty following equation 7 from:
        # https://arxiv.org/pdf/1710.04043.pdf
        for bk_pt in background_pts:
            unary_term[tuple(bk_pt)] = 0 if y_hat[tuple(bk_pt)] == 0 else infty
        
        for fg_pt in foreground_pts:
            unary_term[tuple(fg_pt)] = 0 if y_hat[tuple(fg_pt)] == 1 else infty

        return unary_term

    def _apply_tmp(self, background_pts, foreground_pts, prob):
        # trying the original equation 7 from : https://arxiv.org/pdf/1710.04043.pdf
        # this does not seem to be compatible with MONAI's crf layer as this poses the
        # problem as minimisation problem, whereas MONAI's crf layers solves it as 
        # maximisation problem
        # get infty
        infty = -np.log(max(0.00001, np.min(prob))) * self.scale_infty

        # get label y^
        # FIXME: find an elegant way of "keeping_dims" for argmax
        y_hat = np.expand_dims(np.argmax(prob, axis=self.channel_dim), axis=self.channel_dim)

        unary_term = np.copy(prob)

        # override unary with 0 or infty following equation 7 from:
        # https://arxiv.org/pdf/1710.04043.pdf
        for bk_pt in background_pts:
            unary_term[tuple(bk_pt)] = infty if y_hat[tuple(bk_pt)] == 0 else 0
        
        for fg_pt in foreground_pts:
            unary_term[tuple(fg_pt)] = infty if y_hat[tuple(fg_pt)] == 1 else 0

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

        # extract background/foreground points from image
        # help from: https://stackoverflow.com/a/58087561
        background_pts = np.argwhere(scrib == self.sc_background_label)
        foregroud_pts = np.argwhere(scrib ==  self.sc_foreground_label)
        print(background_pts[0])
        d[self.unary] = self._apply(background_pts, foregroud_pts, prob)
        # d[self.unary] = self._apply_tmp(background_pts, foregroud_pts, prob)
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
          self.simplecrf_params['PosW'] = gaussian_weight #2.0
          self.simplecrf_params['PosRStd'] = gaussian_spatial_sigma #5
          self.simplecrf_params['PosCStd'] = 5
          self.simplecrf_params['PosZStd'] = 5
          self.simplecrf_params['BilateralW'] = bilateral_weight #3.0
          self.simplecrf_params['BilateralRStd'] = bilateral_spatial_sigma #5.0
          self.simplecrf_params['BilateralCStd'] = bilateral_color_sigma #5.0
          self.simplecrf_params['BilateralZStd'] = 5.0
          self.simplecrf_params['ModalityNum'] = 1
          self.simplecrf_params['BilateralModsStds'] = (5.0,)
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

            print(unary_term.shape)
            print(pairwise_term.shape)

            if isinstance(unary_term, torch.Tensor):
                unary_term = unary_term.cpu().numpy()
            
            if isinstance(pairwise_term, torch.Tensor):
                pairwise_term = pairwise_term.cpu().numpy()

            unary_term = np.transpose(np.squeeze(unary_term, axis=0), (1, 2, 3, 0)) # channel last + squeeze
            pairwise_term = np.transpose(np.squeeze(pairwise_term, axis=0), (1, 2, 3, 0)) # channel last + squeeze
        
            print(np.unique(pairwise_term))
            min_p = pairwise_term.min()
            max_p = pairwise_term.max()
            pairwise_term = (((pairwise_term-min_p)/(max_p-min_p)) * 255).astype(np.uint8)
            print(np.unique(pairwise_term))

            d[self.post_proc_label] = denseCRF3D.densecrf3d(pairwise_term, unary_term, self.simplecrf_params).astype(np.float64)

            print(d[self.post_proc_label].dtype)
            print(np.unique(d[self.post_proc_label]))

            orig_label = np.argmax(d['logits'], axis=0)

            print(np.allclose(orig_label, d[self.post_proc_label]))
        else:
            unary_term = d[self.unary].to(self.device)
            pairwise_term = d[self.pairwise].to(self.device)
            
            print(unary_term.dtype)
            print(unary_term.shape)
            # print(torch.sum(unary_term, dim=1))
            print()
            print(pairwise_term.dtype)
            print(pairwise_term.shape)
            print()

            # np.save('unary.npy', unary_term.cpu().detach().numpy())
            # np.save('pairwise.npy', pairwise_term.cpu().detach().numpy())

            d[self.post_proc_label] = torch.argmax(self.crf_layer(unary_term, pairwise_term), dim=1, keepdims=True)
        return d