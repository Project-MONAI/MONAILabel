import logging
import json
import numpy as np
from typing import Tuple

from monai.transforms.transform import Randomizable, Transform
from monai.utils import optional_import

distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")

logger = logging.getLogger(__name__)

# You can write your transforms here... which can be used in your train/infer tasks

class CustomAddRandomGuidanced(Randomizable, Transform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.
    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key that represents discrepancies found between label and prediction, shape (2, C, D, H, W) or (2, C, H, W)
        probability: key that represents click/interaction probability, shape (1)
        fn_fp_click_ratio: ratio of clicks between FN and FP  
    """

    def __init__(
        self,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        fn_fp_click_ratio: Tuple[float, float] = (1.0, 1.0),
    ):
        self.guidance = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self.fn_fp_click_ratio = fn_fp_click_ratio
        self._will_interact = None

    def randomize(self, data=None):
        probability = data[self.probability]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy):
        distance = distance_transform_cdt(discrepancy).flatten()
        probability = np.exp(distance) - 1.0
        idx = np.where(discrepancy.flatten() > 0)[0]

        if np.sum(discrepancy > 0) > 0:
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            return g
        return None

    def add_guidance(self, discrepancy, will_interact):
        if not will_interact:
            return None, None

        pos_discr = discrepancy[0]
        neg_discr = discrepancy[1]

        can_be_positive = np.sum(pos_discr) > 0
        can_be_negative = np.sum(neg_discr) > 0
        
        pos_prob = self.fn_fp_click_ratio[0]/(self.fn_fp_click_ratio[0] + self.fn_fp_click_ratio[1])
        neg_prob = self.fn_fp_click_ratio[1]/(self.fn_fp_click_ratio[0] + self.fn_fp_click_ratio[1])
        
        correct_pos = self.R.choice([True, False], p=[pos_prob, neg_prob])
        
        if can_be_positive and not can_be_negative:
            return self.find_guidance(pos_discr), None
        
        if not can_be_positive and can_be_negative:
            return None, self.find_guidance(neg_discr)
        
        if correct_pos and can_be_positive:
            return self.find_guidance(pos_discr), None 

        if not correct_pos and can_be_negative:
            return None, self.find_guidance(neg_discr)
        return None, None

    def _apply(self, guidance, discrepancy):
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
        pos, neg = self.add_guidance(discrepancy, self._will_interact)
        if pos:
            guidance[0].append(pos)
            guidance[1].append([-1] * len(pos))
        if neg:
            guidance[0].append([-1] * len(neg))
            guidance[1].append(neg)

        return json.dumps(np.asarray(guidance).astype(int).tolist())

    def __call__(self, data):
        d = dict(data)
        guidance = d[self.guidance]
        discrepancy = d[self.discrepancy]
        self.randomize(data)
        d[self.guidance] = self._apply(guidance, discrepancy)
        return d