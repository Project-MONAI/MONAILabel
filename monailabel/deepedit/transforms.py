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
                (len(self.label_names), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32
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
class SelectLabelsAbdomenDatasetd(MapTransform):
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
                new_label_names = {}

                # Making other labels as background
                for k in self.all_label_values.keys():
                    if k not in self.label_names.keys():
                        d[key][d[key] == self.all_label_values[k]] = 0.0

                # Making sure the range values and number of labels are the same
                for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                    if key_label != "background":
                        new_label_names[key_label] = idx
                        d[key][d[key] == val_label] = idx
                    if key_label == "background":
                        new_label_names["background"] = 0
                        d[key][d[key] == self.label_names["background"]] = 0
                d["label_names"] = new_label_names
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
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sigma: int = 2,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch

    def _get_signal(self, image, guidance):
        dimensions = 3 if len(image.shape) > 3 else 2
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
        # In inference the user may not provide clicks for some channels/labels
        if len(guidance):
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
        else:
            if dimensions == 3:
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            else:
                signal = np.zeros((1, image.shape[-2], image.shape[-1]), dtype=np.float32)
            return signal

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                image = d[key]
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]
                guidance = d[self.guidance]
                for key_label in guidance.keys():
                    # Getting signal based on guidance
                    signal = self._get_signal(image, guidance[key_label])
                    tmp_image = np.concatenate([tmp_image, signal], axis=0)
                d[key] = tmp_image
                # logger.info(
                #     f"Number of input channels: {d[key].shape[0]} - "
                #     f'Using image: {d["image_meta_dict"]["filename_or_obj"].split("/")[-1]}'
                # )
                return d
            else:
                print("This transform only applies to image key")
        return d


class FindAllValidSlicesCustomMultiLabeld(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CDHW, where C=1.

    Args:
        label: key to the label source.
        sids: key to store slices indices having valid label map.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sids="sids",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids

    def _apply(self, label, d):
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[1]):  # Assume channel is first
                if d["label_names"][key_label] in label[0][sid]:
                    l_ids.append(sid)
            sids[key_label] = l_ids
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

                sids = self._apply(label, d)
                if sids is not None and len(sids.keys()):
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")
        return d


class AddInitialSeedPointCustomMultiLabeld(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.

    Note that the label is of size (C, D, H, W) or (C, H, W)

    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)

    Args:
        label: label source.
        guidance: key to store guidance.
        sids: key that represents lists of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: Dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions

    def _apply(self, label, sid, key_label):
        dimensions = 3 if len(label.shape) > 3 else 2
        self.default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][sid][np.newaxis]  # Assume channel is first

        # import matplotlib.pyplot as plt
        # plt.imshow(label[0])
        # plt.title('label as is')
        # plt.show()
        # plt.close()

        # THERE MAY BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
        label = (label > 0.5).astype(np.float32)
        # measure.label: Label connected regions of an integer array - Two pixels are connected
        # when they are neighbors and have the same value
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label
        if np.max(blobs_labels) <= 0:
            raise AssertionError(f"SLICES NOT FOUND FOR LABEL: {key_label}")

        # plt.imshow(blobs_labels[0])
        # plt.title('Blobs')
        # plt.show()
        # plt.close()

        pos_guidance = []
        for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
            if dims == 2:
                label = (blobs_labels == ridx).astype(np.float32)
                if np.sum(label) == 0:
                    pos_guidance.append(self.default_guidance)
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

        return np.asarray([pos_guidance])

    def _randomize(self, d, key_label):
        sids = d.get(self.sids_key, None).get(key_label, None) if d.get(self.sids_key, None) is not None else None
        sid = d.get(self.sid_key, None).get(key_label, None) if d.get(self.sid_key, None) is not None else None
        if sids is not None and sids:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            logger.info(f"Not slice IDs for label: {key_label}")
            sid = None
        self.sid[key_label] = sid

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label_guidances = {}
                for key_label in d["sids"].keys():
                    # For all non-background labels
                    if key_label != "background" or d["label_names"][key_label] != 0:
                        # Randomize: Select a random slice
                        self._randomize(d, key_label)
                        # Generate guidance base on selected slice
                        tmp_label = np.copy(d[key])
                        # Taking one label to create the guidance
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 0
                        label_guidances[key_label] = json.dumps(
                            self._apply(tmp_label, self.sid.get(key_label, None), key_label).astype(int).tolist()
                        )
                    elif key_label == "background" or d["label_names"][key_label] == 0:
                        label_guidances[key_label] = json.dumps(
                            np.asarray([[self.default_guidance] * self.connected_regions]).astype(int).tolist()
                        )
                d[self.guidance] = label_guidances
                return d
            else:
                print("This transform only applies to label key")
        return d


class FindDiscrepancyRegionsCustomMultiLabeld(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        label: key to label source.
        pred: key to prediction source.
        discrepancy: key to store discrepancies found between label and prediction.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred: str = "pred",
        discrepancy: str = "discrepancy",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred = pred
        self.discrepancy = discrepancy

    @staticmethod
    def disparity(label, pred):
        disparity = label - pred
        # Negative ONES mean predicted label is not part of the ground truth
        # Positive ONES mean predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).astype(np.float32)
        neg_disparity = (disparity < 0).astype(np.float32)
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                all_discrepancies = {}
                for _, (key_label, val_label) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        # Taking single label
                        label = np.copy(d[key])
                        label[label != val_label] = 0
                        # Label should be represented in 1
                        label = (label > 0.5).astype(np.float32)
                        # Taking single prediction
                        # idx = 0 in pred is the background
                        # pred = d[self.pred][idx+1, ...][np.newaxis]
                        pred = np.copy(d[self.pred])
                        pred[pred != val_label] = 0
                        # Prediction should be represented in one
                        pred = (pred > 0.5).astype(np.float32)
                        all_discrepancies[key_label] = self._apply(label, pred)
                d[self.discrepancy] = all_discrepancies
                return d
            else:
                print("This transform only applies to 'label' key")
        return d


class PosNegClickProbAddRandomGuidanceCustomMultiLabeld(Randomizable, MapTransform):
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
        keys: KeysCollection,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        pos_click_probability: float = 0.5,
        weight_map: Optional[dict] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
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

    def add_guidance(self, discrepancy, mask_background, weight_map):

        pos_discr = discrepancy[0]
        neg_discr = discrepancy[1] * mask_background

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

    def _apply(self, guidance, discrepancy, mask_background, guidance_background, weight_map):

        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance

        guidance_background = (
            guidance_background.tolist() if isinstance(guidance_background, np.ndarray) else guidance_background
        )
        guidance_background = (
            json.loads(guidance_background) if isinstance(guidance_background, str) else guidance_background
        )

        pos, neg = self.add_guidance(discrepancy, mask_background, weight_map)

        if pos:
            guidance[0].append(pos)
            # guidance[1].append([-1] * len(pos))
            self.is_pos = True

        if neg:
            # guidance[0].append([-1] * len(neg))
            guidance_background[0].append(neg)
            self.is_neg = True

        return json.dumps(np.asarray(guidance).astype(int).tolist()), json.dumps(
            np.asarray(guidance_background).astype(int).tolist()
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        guidance = d[self.guidance]
        discrepancy = d[self.discrepancy]
        weight_map = d[self.weight_map] if self.weight_map is not None else None
        # Decide whether to add clicks or not
        self.randomize(data)
        if self._will_interact:
            all_is_pos = {}
            all_is_neg = {}
            # Create mask background to multiply for discrepancy
            mask_background = np.copy(d["label"])
            mask_background[mask_background != 0] = 1.0
            mask_background = 1.0 - mask_background
            for key_label in d["label_names"].keys():
                if key_label != "background":
                    # Add POSITIVE and NEGATIVE (background) guidance based on discrepancy
                    d[self.guidance][key_label], d[self.guidance]["background"] = self._apply(
                        guidance[key_label],
                        discrepancy[key_label],
                        mask_background,
                        guidance["background"],
                        weight_map[key_label] if weight_map is not None else weight_map,
                    )
                    all_is_pos[key_label] = self.is_pos
                    all_is_neg[key_label] = self.is_neg
                    self.is_pos = False
                    self.is_neg = False
            d["is_pos"] = all_is_pos
            d["is_neg"] = all_is_neg
        return d


# A transform to get single modality if there are more and do label sanity
class SingleModalityLabelSanityd(MapTransform):
    """
    Gets single modality and perform label sanity check

    Error is the label is not in the same range:
     https://stdworkflow.com/866/runtimeerror-cudnn-error-cudnn-status-not-initialized
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.label_names = label_names

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "label":
                logger.info(f"Input image shape check in SingleModalityLabelSanityd transform: {d[key].shape}")
            if key == "image":
                meta_data = d["image_meta_dict"]
                if meta_data["spatial_shape"].shape[0] > 3:
                    if meta_data["spatial_shape"][4] > 0:
                        logger.info(
                            f"Image {meta_data['filename_or_obj'].split('/')[-1]} has more than one modality "
                            f"- taking FIRST modality ..."
                        )

                        d[key] = d[key][..., 0]
                        meta_data["spatial_shape"][4] = 0.0

        return d


class AddGuidanceFromPointsCustomMultipleLabeld(Transform):
    """
    Add guidance based on user clicks. ONLY WORKS FOR 3D

    We assume the input is loaded by LoadImaged and has the shape of (H, W, D) originally.
    Clicks always specify the coordinates in (H, W, D)

    If depth_first is True:

        Input is now of shape (D, H, W), will return guidance that specifies the coordinates in (D, H, W)

    else:

        Input is now of shape (H, W, D), will return guidance that specifies the coordinates in (H, W, D)

    Args:
        ref_image: key to reference image to fetch current and original image details.
        guidance: output key to store guidance.
        foreground: key that represents user foreground (+ve) clicks.
        background: key that represents user background (-ve) clicks.
        axis: axis that represents slices in 3D volume. (axis to Depth)
        depth_first: if depth (slices) is positioned at first dimension.
        meta_keys: explicitly indicate the key of the meta data dictionary of `ref_image`.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, original_shape, etc.
            if None, will try to construct meta_keys by `{ref_image}_{meta_key_postfix}`.
        meta_key_postfix: if meta_key is None, use `{ref_image}_{meta_key_postfix}` to to fetch the meta data according
            to the key data, default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.

    """

    def __init__(
        self,
        ref_image,
        guidance: str = "guidance",
        foreground: str = "foreground",
        background: str = "background",
        axis: int = 0,
        depth_first: bool = True,
        meta_keys: Optional[str] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        self.ref_image = ref_image
        self.guidance = guidance
        self.foreground = foreground
        self.background = background
        self.axis = axis
        self.depth_first = depth_first
        self.meta_keys = meta_keys
        self.meta_key_postfix = meta_key_postfix

    def _apply(self, clicks, factor):
        if len(clicks):
            guidance = np.multiply(clicks, factor).astype(int).tolist()
            return guidance
        else:
            return []

    def __call__(self, data):
        d = dict(data)
        meta_dict_key = self.meta_keys or f"{self.ref_image}_{self.meta_key_postfix}"
        if meta_dict_key not in d:
            raise RuntimeError(f"Missing meta_dict {meta_dict_key} in data!")
        if "spatial_shape" not in d[meta_dict_key]:
            raise RuntimeError('Missing "spatial_shape" in meta_dict!')
        original_shape = d[meta_dict_key]["spatial_shape"]
        current_shape = list(d[self.ref_image].shape)

        if self.depth_first:
            if self.axis != 0:
                raise RuntimeError("Depth first means the depth axis should be 0.")
            # in here we assume the depth dimension was in the last dimension of "original_shape"
            original_shape = np.roll(original_shape, 1)

        factor = np.array(current_shape) / original_shape

        fg_bg_clicks = dict()
        # For foreground clicks
        for key_label in d[self.foreground]:
            clicks = d[self.foreground][key_label]
            clicks = list(np.array(clicks).astype(int))
            if self.depth_first:
                for i in range(len(clicks)):
                    clicks[i] = list(np.roll(clicks[i], 1))
            fg_bg_clicks[key_label] = clicks
        # For background clicks
        clicks = d[self.background]
        clicks = list(np.array(clicks).astype(int))
        if self.depth_first:
            for i in range(len(clicks)):
                clicks[i] = list(np.roll(clicks[i], 1))
        fg_bg_clicks["background"] = clicks
        # Creating guidance based on foreground clicks
        all_guidances = dict()
        for key_label in d[self.foreground]:
            all_guidances[key_label] = self._apply(fg_bg_clicks[key_label], factor)
        # Creating guidance based on background clicks
        all_guidances["background"] = self._apply(fg_bg_clicks["background"], factor)
        d[self.guidance] = all_guidances
        return d


class ResizeGuidanceMultipleLabelCustomd(Transform):
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
        all_guidances = dict()
        for key_label in d[self.guidance].keys():
            guidance = (
                np.multiply(d[self.guidance][key_label], factor).astype(int).tolist()
                if len(d[self.guidance][key_label])
                else []
            )
            all_guidances[key_label] = guidance

        d[self.guidance] = all_guidances
        return d


class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation

    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "pred":
                for idx, (key_label, _) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
                        d[f"label_{key_label}"] = d["label"][idx, ...][None]
            elif key != "pred":
                logger.info("This is only for pred key")
        return d


class ToCheckTransformd(MapTransform):
    """
    Transform to debug dictionary

    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            logger.info(f"Printing pred shape in ToCheckTransformd: {d[key].shape}")
        return d
