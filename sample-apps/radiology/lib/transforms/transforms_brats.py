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
import copy
from typing import Dict

from monai.config import KeysCollection
from monai.transforms import MapTransform


class GetSingleModalityBRATSd(MapTransform):
    """
    Gets one modality

    "0": "FLAIR",
    "1": "T1w",
    "2": "t1gd",
    "3": "T2w"

    """

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                # TRANSFORM IN PROGRESS - SHOULD AFFINE AND ORIGINAL BE CHANGED??

                # Output is only one channel
                # Get T1 Gadolinium. Better to describe brain tumour. FLAIR is better for edema (swelling in the brain)
                d[key] = d[key][..., 2]
                # d['image_meta_dict']["original_affine"] = d['image_meta_dict']["original_affine"][0, ...]
                # d['image_meta_dict']["affine"] = d['image_meta_dict']["affine"][0, ...]
                # d[key] = d[key][None] # Add dimension
            else:
                print("Transform 'GetSingleModalityBRATSd' only works for the image key")

        return d


class MaskTumord(MapTransform):
    """
    Mask the tumor and edema labels

    """

    def __call__(self, data):
        d: Dict = dict(data)
        mask_tumor_edema = copy.deepcopy(d["label"])
        mask_tumor_edema[mask_tumor_edema < 15] = 1
        mask_tumor_edema[mask_tumor_edema > 1] = 0
        for key in self.key_iterator(d):
            if len(d[key].shape) == 4:
                for i in range(d[key].shape[-1]):
                    d[key][..., i] = d[key][..., i] * mask_tumor_edema
            else:
                d[key] = d[key] * mask_tumor_edema
        return d


class MergeLabelsd(MapTransform):
    """
    Merge conflicting labels into single one

    """

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            merged_labels = copy.deepcopy(d[key])
            merged_labels[merged_labels == 6] = 6  # Thalamus
            merged_labels[merged_labels == 7] = 6  # Caudate
            merged_labels[merged_labels == 8] = 8  # Putamen
            merged_labels[merged_labels == 9] = 8  # Pallidum
            d[key] = merged_labels
        return d


class AddUnknownLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, max_labels=None, allow_missing_keys: bool = False):
        """
        Assign unknown label to intensities bigger than 0 - background is anything else

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.max_labels = max_labels

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            unknown_mask = copy.deepcopy(d["image"][..., 0])
            unknown_mask[unknown_mask > unknown_mask.max() * 0.10] = 1
            mask_all_labels = copy.deepcopy(d[key])
            mask_all_labels[mask_all_labels > 0] = 1
            unknown_mask = unknown_mask - mask_all_labels
            unknown_mask[unknown_mask < 0] = 0
            unknown_mask[unknown_mask == 1] = self.max_labels
            d[key] = d[key] + unknown_mask
        return d
