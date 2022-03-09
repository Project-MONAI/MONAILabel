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
import pathlib

import numpy as np
import openslide
from monai.apps.deepgrow.transforms import AddInitialSeedPointd
from monai.config import KeysCollection
from monai.transforms import CenterSpatialCrop, MapTransform, Transform
from PIL import Image
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)


class LoadImagePatchd(MapTransform):
    def __init__(
        self, keys: KeysCollection, meta_key_postfix: str = "meta_dict", conversion="RGB", dtype=np.uint8, padding=True
    ):
        super().__init__(keys)
        self.meta_key_postfix = meta_key_postfix
        self.conversion = conversion
        self.dtype = dtype
        self.padding = padding

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], str):
                continue  # Support direct image in np (pass only transform)

            name = d[key]
            ext = pathlib.Path(name).suffix
            if ext == ".npy":
                d[key] = np.load(d[key])
                continue

            location = d.get("location", (0, 0))
            level = d.get("level", 0)
            size = d.get("size", None)

            # Model input size
            tile_size = d.get("tile_size", size)

            if not ext or ext in (
                ".bif",
                ".mrxs",
                ".ndpi",
                ".scn",
                ".svs",
                ".svslide",
                ".tif",
                ".tiff",
                ".vms",
                ".vmu",
            ):
                slide = openslide.OpenSlide(name)
                size = size if size else slide.dimensions
                img = slide.read_region(location, level, size)
            else:
                img = Image.open(d[key])

            img = img.convert(self.conversion)
            image_np = np.array(img, dtype=self.dtype)

            meta_dict_key = f"{key}_{self.meta_key_postfix}"
            meta_dict = d.get(meta_dict_key)
            if meta_dict is None:
                d[meta_dict_key] = dict()
                meta_dict = d.get(meta_dict_key)

            meta_dict["spatial_shape"] = np.asarray(image_np.shape[:-1])
            meta_dict["original_channel_dim"] = -1
            logger.debug(f"Image shape: {image_np.shape} vs size: {size} vs tile_size: {tile_size}")

            if self.padding and image_np.shape[0] != tile_size[0] or image_np.shape[1] != tile_size[1]:
                image_padded = np.zeros((tile_size[0], tile_size[1], 3), dtype=image_np.dtype)
                image_padded[0 : image_np.shape[0], 0 : image_np.shape[1]] = image_np
                image_np = image_padded
            d[key] = image_np

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


def mask_percent(img_np):
    if (len(img_np.shape) == 3) and (img_np.shape[2] == 3):
        np_sum = img_np[:, :, 0] + img_np[:, :, 1] + img_np[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(img_np) / img_np.size * 100
    return mask_percentage


def filter_green_channel(img_np, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    g = img_np[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        gr_ch_mask = filter_green_channel(img_np, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    return gr_ch_mask


def filter_grays(rgb, tolerance=15):
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    return ~(rg_diff & rb_diff & gb_diff)


def filter_ostu(img):
    mask = np.dot(img[..., :3], [0.2125, 0.7154, 0.0721]).astype(np.uint8)
    mask = 255 - mask
    return mask > threshold_otsu(mask)


def filter_remove_small_objects(img_np, min_size=3000, avoid_overmask=True, overmask_thresh=95):
    rem_sm = remove_small_objects(img_np.astype(bool), min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = round(min_size / 2)
        rem_sm = filter_remove_small_objects(img_np, new_min_size, avoid_overmask, overmask_thresh)
    return rem_sm


class FilterImaged(MapTransform):
    def __init__(self, keys: KeysCollection, min_size=500):
        super().__init__(keys)
        self.min_size = min_size

    def filter(self, rgb):
        mask_not_green = filter_green_channel(rgb)
        mask_not_gray = filter_grays(rgb)
        mask_gray_green = mask_not_gray & mask_not_green
        mask = (
            filter_remove_small_objects(mask_gray_green, min_size=self.min_size) if self.min_size else mask_gray_green
        )

        return rgb * np.dstack([mask, mask, mask])

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self.filter(img)
        return d


class PostFilterLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, image="image", min_size=80):
        super().__init__(keys)
        self.image = image
        self.min_size = min_size

    def __call__(self, data):
        d = dict(data)
        img = d[self.image]
        img = img[:3]
        img = np.moveaxis(img, 0, -1)  # to channel last
        img = img * 128 + 128
        img = img.astype(np.uint8)

        for key in self.keys:
            label = d[key].astype(np.uint8)
            gray = np.dot(img, [0.2125, 0.7154, 0.0721])
            label = label * np.logical_xor(label, gray == 0)
            label = filter_remove_small_objects(label, min_size=self.min_size).astype(np.uint8)
            d[key] = label
        return d


class AddInitialSeedPointExd(AddInitialSeedPointd):
    def _apply(self, label, sid):
        try:
            return super()._apply(label, sid)
        except AssertionError:
            dimensions = 2
            default_guidance = [-1] * (dimensions + 1)
            return np.asarray([[default_guidance], [default_guidance]])


class AddClickGuidanced(Transform):
    def __init__(
        self,
        image,
        guidance="guidance",
        foreground="foreground",
        background="background",
    ):
        self.image = image
        self.guidance = guidance
        self.foreground = foreground
        self.background = background

    def __call__(self, data):
        d = dict(data)

        wsi_meta = d.get("wsi", {})
        location = wsi_meta.get("location", (0, 0))
        tx, ty = location[0], location[1]

        pos = d.get(self.foreground)
        pos = (np.array(pos) - (tx, ty)).astype(int).tolist() if pos else []

        neg = d.get(self.background)
        neg = (np.array(neg) - (tx, ty)).astype(int).tolist() if neg else []

        d[self.guidance] = [pos, neg]
        return d
