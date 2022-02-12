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
import math

import numpy as np
from PIL import Image
from monai.config import KeysCollection
from monai.transforms import CenterSpatialCrop, MapTransform
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import remove_small_objects

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

    def mask_percent(self, img_np):
        if (len(img_np.shape) == 3) and (img_np.shape[2] == 3):
            np_sum = img_np[:, :, 0] + img_np[:, :, 1] + img_np[:, :, 2]
            mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
        else:
            mask_percentage = 100 - np.count_nonzero(img_np) / img_np.size * 100
        return mask_percentage

    def filter_green_channel(self, img_np, green_thresh=200, avoid_overmask=True, overmask_thresh=90,
                             output_type="bool"):
        g = img_np[:, :, 1]
        gr_ch_mask = (g < green_thresh) & (g > 0)
        mask_percentage = self.mask_percent(gr_ch_mask)
        if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
            new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
            gr_ch_mask = self.filter_green_channel(img_np, new_green_thresh, avoid_overmask, overmask_thresh,
                                                   output_type)
        return gr_ch_mask

    def filter_grays(self, rgb, tolerance=15):
        rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
        rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
        gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
        return ~(rg_diff & rb_diff & gb_diff)

    def filter_ostu(self, img):
        mask = np.dot(img[..., :3], [0.2125, 0.7154, 0.0721]).astype(np.uint8)
        mask = 255 - mask
        return mask > threshold_otsu(mask)

    def filter_remove_small_objects(self, img_np, min_size=3000, avoid_overmask=True, overmask_thresh=95):
        rem_sm = remove_small_objects(img_np, min_size=min_size)
        mask_percentage = self.mask_percent(rem_sm)
        if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
            new_min_size = round(min_size / 2)
            rem_sm = self.filter_remove_small_objects(img_np, new_min_size, avoid_overmask, overmask_thresh)
        return rem_sm

    def filter(self, rgb):
        mask_not_green = self.filter_green_channel(rgb)
        mask_not_gray = self.filter_grays(rgb)
        mask_gray_green = mask_not_gray & mask_not_green
        mask = self.filter_remove_small_objects(mask_gray_green, min_size=500)

        return rgb * np.dstack([mask, mask, mask])

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self.filter(img)
        return d


class NormalizeImaged(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def normalize(self, img):
        img = (img - 128.0) / 128.0
        return img.astype(np.float32)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self.normalize(img)
        return d
