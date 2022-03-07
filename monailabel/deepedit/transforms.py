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
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
from monai.config import KeysCollection
from monai.transforms import CropForeground, ResizeWithPadOrCrop
from monai.transforms.transform import MapTransform, Randomizable, Transform

logger = logging.getLogger(__name__)

from monai.utils import optional_import

distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


class DiscardAddGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        number_intensity_ch: int = 1,
        probability: float = 1.0,
        allow_missing_keys: bool = False,
    ):
        """
        Discard positive and negative points according to discard probability

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        :param number_intensity_ch: number of intensity channels
        :param probability: probability of discarding clicks
        """
        super().__init__(keys, allow_missing_keys)

        self.number_intensity_ch = number_intensity_ch
        self.discard_probability = probability

    def _apply(self, image):
        if self.discard_probability >= 1.0 or np.random.choice(
            [True, False], p=[self.discard_probability, 1 - self.discard_probability]
        ):
            signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            if image.shape[0] == self.number_intensity_ch + 2:
                image[self.number_intensity_ch] = signal
                image[self.number_intensity_ch + 1] = signal
            else:
                image = np.concatenate((image, signal, signal), axis=0)
        return image

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                d[key] = self._apply(d[key])
            else:
                print("This transform only applies to the image")
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

        factor = np.divide(current_shape, d["image_meta_dict"]["spatial_shape"])
        pos_clicks, neg_clicks = d["foreground"], d["background"]

        pos = np.multiply(pos_clicks, factor).astype(int).tolist() if len(pos_clicks) else []
        neg = np.multiply(neg_clicks, factor).astype(int).tolist() if len(neg_clicks) else []

        d[self.guidance] = [pos, neg]
        return d


class AddRandomGuidanced(Randomizable, Transform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key to discrepancy map between label and prediction
          shape (2, C, H, W, D) or (2, C, H, W)
        probability: key to click/interaction probability, shape (1)
        weight_map: optional key to predetermined weight map used to increase click likelihood
          in higher weight areas shape (C, H, W, D) or (C, H, W)
    """

    def __init__(
        self,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        weight_map: Optional[str] = None,
        probability: str = "probability",
    ):
        self.guidance = guidance
        self.discrepancy = discrepancy
        self.weight_map = weight_map
        self.probability = probability
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

        if np.sum(probability[idx]) > 0:
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

        correct_pos = np.sum(pos_discr) >= np.sum(neg_discr)

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
        if neg:
            guidance[0].append([-1] * len(neg))
            guidance[1].append(neg)
            self.is_neg = True
        return json.dumps(np.asarray(guidance).astype(int).tolist())

    def __call__(self, data):
        d = dict(data)
        guidance = d[self.guidance]
        discrepancy = d[self.discrepancy]
        weight_map = d[self.weight_map] if self.weight_map is not None else None
        self.randomize(data)
        d[self.guidance] = self._apply(guidance, discrepancy, weight_map)
        d["is_pos"] = self.is_pos
        d["is_neg"] = self.is_neg
        self.is_pos = False
        self.is_neg = False
        return d


class PosNegClickProbAddRandomGuidanced(Randomizable, Transform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability: key to click/interaction probability, shape (1)
        pos_click_probability: if click, probability of a positive click
          (probability of negative click will be 1 - pos_click_probability)
        weight_map: optional key to predetermined weight map used to increase click likelihood
          in higher weight areas shape (C, H, W, D) or (C, H, W)
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

        if np.sum(probability[idx]) > 0:
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
        if neg:
            guidance[0].append([-1] * len(neg))
            guidance[1].append(neg)
            self.is_neg = True
        return json.dumps(np.asarray(guidance).astype(int).tolist())

    def __call__(self, data):
        d = dict(data)
        guidance = d[self.guidance]
        discrepancy = d[self.discrepancy]
        weight_map = d[self.weight_map] if self.weight_map is not None else None
        self.randomize(data)
        d[self.guidance] = self._apply(guidance, discrepancy, weight_map)
        d["is_pos"] = self.is_pos
        d["is_neg"] = self.is_neg
        self.is_pos = False
        self.is_neg = False
        return d


class CropGuidanceForegroundd(Transform):
    """
    Update guidance based on foreground crop.
    Transform should precede CropForegroundd applied to image, in transforms list.
    Args:
        ref_image: reference image key.
        guidance: guidance key.
        source_key: mask key for generating bounding box.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """

    def __init__(self, ref_image: str, guidance: str, source_key: str, margin: Union[Sequence[int], int] = 0) -> None:
        self.ref_image = ref_image
        self.guidance = guidance
        self.source_key = source_key
        self.margin = margin

    def __call__(self, data):
        d = dict(data)

        current_shape = d[self.ref_image].shape[1:]
        dims = len(current_shape)
        cropper = CropForeground(margin=self.margin)

        # get guidance following foreground crop
        new_guidance = []
        for guidance in d[self.guidance]:
            if guidance:
                signal = np.zeros(current_shape)
                for point in guidance:
                    if dims == 2:
                        signal[point[0], point[1]] = 1.0
                    else:
                        signal[point[0], point[1], point[2]] = 1.0
                # compute bounding box
                box_start, box_end = cropper.compute_bounding_box(img=d[self.source_key])
                signal = (
                    cropper.crop_pad(img=signal[np.newaxis, :], box_start=box_start, box_end=box_end).squeeze(0),
                )  # requires channel dim
                new_guidance.append(
                    np.argwhere(signal[0] == 1.0).astype(int).tolist()
                )  # signal is a tuple containing a numpy array
            else:
                new_guidance.append([])

        d[self.guidance] = new_guidance
        return d


class ResizeGuidanceWithPadOrCropd(Transform):
    """
    Update guidance based on pad or crop.
    Transform should precede ResizeWithPadOrCropd applied to image, in transforms list.
    Args:
        ref_image: reference image key.
        guidance: guidance key.
        spatial_size: the spatial size of output data after pad or crop.
    """

    def __init__(self, ref_image: str, guidance: str, spatial_size: Sequence[int]) -> None:
        self.ref_image = ref_image
        self.guidance = guidance
        self.spatial_size = spatial_size

    def __call__(self, data):
        d = dict(data)

        current_shape = d[self.ref_image].shape[1:]
        dims = len(current_shape)
        croppad = ResizeWithPadOrCrop(spatial_size=self.spatial_size)

        # get guidance following pad or crop to spatial_size
        new_guidance = []
        for guidance in d[self.guidance]:
            if guidance:
                signal = np.zeros(current_shape)
                for point in guidance:
                    if dims == 2:
                        signal[point[0], point[1]] = 1.0
                    else:
                        signal[point[0], point[1], point[2]] = 1.0
                signal = croppad(signal[np.newaxis, :]).squeeze(0)  # croppad requires channel dim
                new_guidance.append(np.argwhere(signal == 1.0).astype(int).tolist())
            else:
                new_guidance.append([])

        d[self.guidance] = new_guidance
        return d
