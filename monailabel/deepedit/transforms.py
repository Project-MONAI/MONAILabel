# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Dict, Optional

import numpy as np
from monai.transforms.transform import Randomizable, Transform

logger = logging.getLogger(__name__)

from monai.utils import optional_import

distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")
    
class DiscardAddGuidanced(Transform):
    def __init__(self, image: str = "image", number_intensity_ch: int = 1, discard_probability: float = 1.0):
        """
        Discard positive and negative points according to discard probability
        :param image: image key
        :param number_intensity_ch: number of intensity channels
        :param discard_probability: probability of discarding clicks
        """
        self.image = image
        self.number_intensity_ch = number_intensity_ch
        self.discard_probability = discard_probability

    def _apply(self, image):
        if self.discard_probability >= 1.0 or np.random.choice([True, False], p=[self.discard_probability, 1 - self.discard_probability]):
            signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            if image.shape[0] == self.number_intensity_ch + 2:
                image[self.number_intensity_ch] = signal
                image[self.number_intensity_ch + 1] = signal
            else:
                image = np.concatenate((image, signal, signal), axis=0)
        return image

    def __call__(self, data):
        d: Dict = dict(data)
        d[self.image] = self._apply(d[self.image])
        return d

class ResizeGuidanceCustomd(Transform):
    """
    Resize the guidance based on cropped vs resized image.
    """

    def __init__(
        self,
        guidance: str,
        ref_image: str,
    ) -> None:
        self.guidance = guidance
        self.ref_image = ref_image

    def __call__(self, data):
        d = dict(data)
        current_shape = d[self.ref_image].shape[1:]

        factor = np.divide(current_shape, d["image_meta_dict"]["dim"][1:4])
        pos_clicks, neg_clicks = d["foreground"], d["background"]

        pos = np.multiply(pos_clicks, factor).astype(int).tolist() if len(pos_clicks) else []
        neg = np.multiply(neg_clicks, factor).astype(int).tolist() if len(neg_clicks) else []

        d[self.guidance] = [pos, neg]
        return d


class PosNegClickProbAddRandomGuidanced(Randomizable, Transform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.
    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key that represents discrepancies found between label and prediction, 
            shape (2, C, D, H, W) or (2, C, H, W)
        probability: key that represents click/interaction probability, shape (1)
        pos_click_probability: if click, probability of a positive click 
            (probability of negative click will be 1 - pos_click_probability)
        weight_map: optional key to predetermined weight map used to increase click likelihood in higher weight areas, 
            shape (C, D, H, W) or (C, H, W)
    """

    def __init__(
        self,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        pos_click_probability: float = 0.5,
        weight_map: Optional[str] = None,
    ):
        self.guidance = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self.pos_click_probability = pos_click_probability
        self.weight_map = weight_map
        self._will_interact = None
        self.is_pos = False
        self.is_neg = False

    def randomize(self, data=None):
        probability = data[self.probability]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy, weight_map):
        distance = distance_transform_cdt(discrepancy)
        weighted_distance = (distance * weight_map).flatten() if weight_map is not None else distance.flatten()
        probability = np.exp(weighted_distance) - 1.0
        idx = np.where(discrepancy.flatten() > 0)[0]

        if np.sum(discrepancy > 0) > 0:
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = weighted_distance[seed]

            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            return g
        return None

    def add_guidance(self, discrepancy, weight_map, will_interact):
        if not will_interact:
            return None, None

        pos_discr = discrepancy[0]
        neg_discr = discrepancy[1]

        can_be_positive = np.sum(pos_discr) > 0
        can_be_negative = np.sum(neg_discr) > 0

        pos_prob = self.pos_click_probability
        neg_prob = 1 - pos_prob

        correct_pos = self.R.choice([True, False], p=[pos_prob, neg_prob])

        if can_be_positive and not can_be_negative:
            return self.find_guidance(pos_discr, weight_map), None

        if not can_be_positive and can_be_negative:
            return None, self.find_guidance(neg_discr, weight_map)

        if correct_pos and can_be_positive:
            return self.find_guidance(pos_discr, weight_map), None

        if not correct_pos and can_be_negative:
            return None, self.find_guidance(neg_discr, weight_map)
        return None, None

    def _apply(self, guidance, discrepancy, weight_map):
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
        pos, neg = self.add_guidance(discrepancy, weight_map, self._will_interact)
        if pos:
            guidance[0].append(pos)
            guidance[1].append([-1] * len(pos))
            self.is_pos = True
            print("pos_click")
        elif neg:
            guidance[0].append([-1] * len(neg))
            guidance[1].append(neg)
            self.is_neg = True
            print("neg_click")
        else:
            print("no_click")

        return json.dumps(np.asarray(guidance).astype(int).tolist())

    def __call__(self, data):
        d = dict(data)
        guidance = d[self.guidance]
        discrepancy = d[self.discrepancy]
        weight_map =  d[self.weight_map] if self.weight_map is not None else None
        self.randomize(data)
        d[self.guidance] = self._apply(guidance, discrepancy, weight_map)
        d["is_pos"] = self.is_pos
        d["is_neg"] = self.is_neg
        self.is_pos = False
        self.is_neg = False
        return d
