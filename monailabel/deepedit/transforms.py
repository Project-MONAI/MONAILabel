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
from typing import Dict, Hashable, Mapping, Optional

import numpy as np
import torch
from monai.config import KeysCollection
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import MapTransform, Randomizable, Transform
from skimage import measure

logger = logging.getLogger(__name__)

from monai.utils import optional_import

distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


class DiscardAddGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        number_intensity_ch: int = 1,
        probability: float = 1.0,
        label_names=None,
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
        self.label_names = label_names

    def _apply(self, image):
        if self.discard_probability >= 1.0 or np.random.choice(
            [True, False], p=[self.discard_probability, 1 - self.discard_probability]
        ):
            signal = np.zeros(
                (len(self.label_names) + 1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32
            )
            if image.shape[0] == self.number_intensity_ch + len(self.label_names) + 1:
                image[self.number_intensity_ch :, ...] = signal
            else:
                image = np.concatenate([image, signal], axis=0)
        return image

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                d[key] = self._apply(d[key])
            else:
                print("This transform only applies to the image")
        return d


class DiscardAddGuidanceSingleLabeld(MapTransform):
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


# A transform to get single modality and single label
class SingleLabelSingleModalityd(MapTransform):
    """
    Gets single modality and single label
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "label":
                meta_data = d["label_meta_dict"]
                if d[key].max() > 1:
                    logger.info(
                        f"Label {meta_data['filename_or_obj'].split('/')[-1]} has more than one mask - "
                        f"taking SINGLE mask ..."
                    )
                    result = []
                    # label bigger than 0 is foreground
                    result.append(d[key] > 0)
                    # label 0 is background
                    result.append(d[key] == 0)
                    d[key] = np.stack(result, axis=0).astype(np.float32)

                    d[key] = d[key][0, ...]

                    meta_data["pixdim"][4] = 0.0
                    meta_data["dim"][0] = 3
                    meta_data["dim"][4] = 1

            if key == "image":
                meta_data = d["image_meta_dict"]
                if meta_data["pixdim"][4] > 0:
                    logger.info(
                        f"Image {meta_data['filename_or_obj'].split('/')[-1]} has more than one modality "
                        f"- taking FIRST modality ..."
                    )

                    d[key] = d[key][..., 0]

                    meta_data["pixdim"][4] = 0.0
                    meta_data["dim"][0] = 3
                    meta_data["dim"][4] = 1

        return d


# Transform for multilabel DeepEdit segmentation
class SelectLabelsAbdomend(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        """
        Select labels from list on the Multi-Atlas Labeling Beyond the Cranial Vault dataset

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        :param label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names
        self.all_label_values = {
            "spleen": 1,
            "right_kidney": 2,
            "left_kidney": 3,
            "gallbladder": 4,
            "esophagus": 5,
            "liver": 6,
            "stomach": 7,
            "aorta": 8,
            "inferior_vena_cava": 9,
            "portal_vein": 10,
            "splenic_vein": 11,
            "pancreas": 12,
            "right_adrenal_gland": 13,
            "left_adrenal_gland": 14,
        }

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                # Making other labels as background
                for k in self.all_label_values.keys():
                    if k not in self.label_names:
                        d[key][d[key] == self.all_label_values[k]] = 0.0
            else:
                print("This transform only applies to the label")
        return d


# One label at a time transform - DeepEdit
class SingleLabelSelectiond(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        """
        Selects one label at a time to train the DeepEdit

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        :param label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names
        self.all_label_values = {
            "spleen": 1,
            "right_kidney": 2,
            "left_kidney": 3,
            "gallbladder": 4,
            "esophagus": 5,
            "liver": 6,
            "stomach": 7,
            "aorta": 8,
            "inferior_vena_cava": 9,
            "portal_vein": 10,
            "splenic_vein": 11,
            "pancreas": 12,
            "right_adrenal_gland": 13,
            "left_adrenal_gland": 14,
        }

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                # Taking one label at a time
                t_label = np.random.choice(self.label_names)
                d["current_label"] = t_label
                d[key][d[key] != self.all_label_values[t_label]] = 0.0
                # Convert label to index values following label_names argument
                max_label_val = self.label_names.index(t_label) + 1
                d[key][d[key] > 0] = max_label_val
                print(f"Using label {t_label} with number: {d[key].max()}")
            else:
                print("This transform only applies to the label")
        return d


class AddGuidanceSignalCustomMultiLabeld(MapTransform):
    """
    Add Guidance signal for input image. Multilabel DeepEdit

    Based on the "guidance" points, apply gaussian to them and add them as new channel for input image.

    Args:
        image: key to the image source.
        guidance: key to store guidance.
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        label_names: list of label names
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sigma: int = 2,
        number_intensity_ch: int = 1,
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.label_names = label_names  # This defines the NUMBER OF channels in the input tensor

    def _get_signal(self, image, guidance):
        dimensions = 3 if len(image.shape) > 3 else 2
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
        if dimensions == 3:
            signal = np.zeros((len(guidance), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        else:
            signal = np.zeros((len(guidance), image.shape[-2], image.shape[-1]), dtype=np.float32)

        sshape = signal.shape
        for i, g_i in enumerate(guidance):
            for point in g_i:
                if np.any(np.asarray(point) < 0):
                    continue

                if dimensions == 3:
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[i, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[i, p1, p2] = 1.0

            if np.max(signal[i]) > 0:
                signal_tensor = torch.tensor(signal[i])
                pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                signal_tensor = signal_tensor.squeeze(0).squeeze(0)
                signal[i] = signal_tensor.detach().cpu().numpy()
                signal[i] = (signal[i] - np.min(signal[i])) / (np.max(signal[i]) - np.min(signal[i]))
        return signal

    def _apply(self, image, guidance, num_channels, idx_guidance):
        signal = self._get_signal(image, guidance)
        image = image[0 : 0 + self.number_intensity_ch, ...]
        empty_signal = np.zeros((num_channels, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        input_tensor = np.concatenate([image, empty_signal], axis=0)
        # Assign positive clicks to label channel
        input_tensor[idx_guidance, ...] = signal[0, ...]
        # Assign negative clicks to the last channel
        input_tensor[-1, ...] = signal[1, ...]
        return input_tensor

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                image = d[key]
                guidance = d[self.guidance]
                for l in guidance.keys():
                    # Check if there is guidance
                    if guidance[l] != -1:
                        d[key] = self._apply(
                            image, guidance[l], len(self.label_names) + 1, self.label_names.index(d["current_label"])
                        )
                return d
            else:
                print("This transform only applies to image key")

    # def __call__(self, data):
    #     d = dict(data)
    #     image = d[self.image]
    #     guidance = d[self.guidance]
    #
    #     d[self.image] = self._apply(
    #         image, guidance, len(self.label_names) + 1, self.label_names.index(d["current_label"])
    #     )
    #     return d


class FindAllValidSlicesCustomMultiLabeld(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CDHW, where C=1.

    Args:
        label: key to the label source.
        sids: key to store slices indices having valid label map.
        label_names: list of label names
    """

    def __init__(
        self,
        keys: KeysCollection,
        sids="sids",
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids
        self.label_names = label_names  # UNUSED - How to associate label names with label numbers?

    def _apply(self, label):
        label_numbers = np.unique(label)[1:]  # Assume background is 0.0 and is the first element
        sids = {}
        for l in label_numbers:
            l_ids = []
            for sid in range(label.shape[1]):  # Assume channel is first
                if l in label[0][sid]:
                    l_ids.append(sid)
            sids[str(int(l))] = l_ids
        return sids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                if label.shape[0] != 1:
                    raise ValueError("Only supports single channel labels!")

                if len(label.shape) != 4:  # only for 3D
                    raise ValueError("Only supports label with shape CDHW!")

                sids = self._apply(label)
                if sids is not None and len(sids.keys()):
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")


class AddInitialSeedPointCustomMultiLabeld(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.

    Note that the label is of size (C, D, H, W) or (C, H, W)

    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)

    Args:
        label: label source.
        guidance: key to store guidance.
        sids: key that represents list of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
        label_names: list of label names
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid = None
        self.guidance = guidance
        self.connected_regions = connected_regions
        self.label_names = label_names  # UNUSED - How to associate label names with label numbers?

    def _apply(self, label, sid):
        dimensions = 3 if len(label.shape) > 3 else 2
        default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][sid][np.newaxis]  # Assume channel is first

        # import matplotlib.pyplot as plt
        # plt.imshow(label[0])
        # plt.title('label as is')
        # plt.show()
        # plt.close()

        # REMEMBER: THERE CAN BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
        label = (label > 0.5).astype(np.float32)
        # measure.label: Label connected regions of an integer array - Two pixels are connected
        # when they are neighbors and have the same value
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label
        if np.max(blobs_labels) <= 0:
            raise AssertionError("Not a valid Label")

        # plt.imshow(blobs_labels[0])
        # plt.title('Blobs')
        # plt.show()
        # plt.close()

        pos_guidance = []
        for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
            if dims == 2:
                label = (blobs_labels == ridx).astype(np.float32)
                if np.sum(label) == 0:
                    pos_guidance.append(default_guidance)
                    continue

            # plt.imshow(label[0])
            # plt.title('Label postprocessed with blob number')
            # plt.show()

            # plt.imshow(distance_transform_cdt(label)[0])
            # plt.title('Transform CDT')
            # plt.show()

            # The distance transform provides a metric or measure of the separation of points in the image.
            # This function calculates the distance between each pixel that is set to off (0) and
            # the nearest nonzero pixel for binary images - http://matlab.izmiran.ru/help/toolbox/images/morph14.html
            distance = distance_transform_cdt(label).flatten()
            probability = np.exp(distance) - 1.0

            idx = np.where(label.flatten() > 0)[0]
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
            g[0] = dst[0]  # for debug
            if dimensions == 2 or dims == 3:
                pos_guidance.append(g)
            else:
                pos_guidance.append([g[0], sid, g[-2], g[-1]])

        return np.asarray([pos_guidance, [default_guidance] * len(pos_guidance)])

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label_guidances = {}
                for l in d["sids"].keys():
                    sids = d["sids"][l]
                    if sids is not None:
                        # Randomize: Select a random slice
                        self.sid = self.R.choice(sids, replace=False)
                        # Generate guidance base on selected slice
                        tmp_label = np.copy(d[key])
                        # Taking one label to create the guidance
                        tmp_label[tmp_label != float(l)] = 0.0
                        label_guidances[l] = json.dumps(self._apply(tmp_label, self.sid).astype(int).tolist())
                d[self.guidance] = label_guidances
                return d
            else:
                print("This transform only applies to label key")
