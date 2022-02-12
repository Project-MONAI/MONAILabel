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

import logging

import numpy as np
from monai.config import KeysCollection
from monai.transforms import CenterSpatialCrop, MapTransform
from PIL import Image
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

logger = logging.getLogger(__name__)


class LoadImagePatchd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(f"Type of {key} is: ")
            img = Image.open(d[key]) if isinstance(d[key], str) else d[key]
            img = np.array(img, dtype=np.uint8)
            d[key] = img
        return d


class LabelToChanneld(MapTransform):
    def __init__(self, keys: KeysCollection, labels):
        super().__init__(keys)
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask = d[key]
            img = np.zeros((len(self.labels), mask.shape[0], mask.shape[1]))

            for count, idx in enumerate(self.labels):
                img[count, mask == idx] = 1
            d[key] = img
        return d


class ClipBorderd(MapTransform):
    def __init__(self, keys: KeysCollection, border=2):
        super().__init__(keys)
        self.border = border

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            roi_size = (img.shape[-2] - self.border * 2, img.shape[-1] - self.border * 2)
            crop = CenterSpatialCrop(roi_size=roi_size)
            d[key] = crop(img)
        return d


class FilterImaged(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            # rgb = img
            # tolerance = 30
            # rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
            # rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
            # gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
            # mask = ~(rg_diff & rb_diff & gb_diff)

            mask = np.dot(img[..., :3], [0.2125, 0.7154, 0.0721]).astype(np.uint8)
            mask = 255 - mask
            mask = mask > threshold_otsu(mask)

            mask = remove_small_objects(mask)
            mask = remove_small_holes(mask)

            img = img * np.dstack([mask, mask, mask])
            d[key] = img
        return d


class FilterImaged(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            # rgb = img
            # tolerance = 30
            # rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
            # rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
            # gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
            # mask = ~(rg_diff & rb_diff & gb_diff)

            mask = np.dot(img[..., :3], [0.2125, 0.7154, 0.0721]).astype(np.uint8)
            mask = 255 - mask
            mask = mask > threshold_otsu(mask)

            mask = remove_small_objects(mask)
            mask = remove_small_holes(mask)

            img = img * np.dstack([mask, mask, mask])
            d[key] = img
        return d


class NormalizeImaged(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            img = (img - 128.0) / 128.0
            d[key] = img.astype(np.float32)
        return d
