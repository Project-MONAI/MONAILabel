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

import cv2
import numpy as np
import openslide
from monai.apps.deepgrow.transforms import AddInitialSeedPointd
from monai.config import KeysCollection
from monai.transforms import CenterSpatialCrop, MapTransform
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

            wsi_meta = d.get("wsi", {})
            location = wsi_meta.get("location", (0, 0))
            level = wsi_meta.get("level", 0)
            size = wsi_meta.get("size", None)

            # Model input size
            patch_size = d.get("patch_size", size)

            name = d[key]
            ext = pathlib.Path(name).suffix
            if ext in (".bif", ".mrxs", ".ndpi", ".scn", ".svs", ".svslide", ".tif", ".tiff", ".vms", ".vmu"):
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
            logger.info(f"Image shape: {image_np.shape} vs size: {size} vs patch_size: {patch_size}")

            if self.padding and image_np.shape[0] != patch_size[0] or image_np.shape[1] != patch_size[1]:
                image_padded = np.zeros((patch_size[0], patch_size[1], 3), dtype=image_np.dtype)
                image_padded[0 : image_np.shape[0], 0 : image_np.shape[1]] = image_np
                image_np = image_padded
            d[key] = image_np

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
    rem_sm = remove_small_objects(img_np, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = round(min_size / 2)
        rem_sm = filter_remove_small_objects(img_np, new_min_size, avoid_overmask, overmask_thresh)
    return rem_sm


class FilterImaged(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def filter(self, rgb):
        mask_not_green = filter_green_channel(rgb)
        mask_not_gray = filter_grays(rgb)
        mask_gray_green = mask_not_gray & mask_not_green
        mask = filter_remove_small_objects(mask_gray_green, min_size=500)

        return rgb * np.dstack([mask, mask, mask])

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self.filter(img)
        return d


class NormalizeImaged(MapTransform):
    def normalize(self, img):
        img = (img - 128.0) / 128.0
        return img.astype(np.float32)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self.normalize(img)
        return d


class PostFilterLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, image="image"):
        super().__init__(keys)
        self.image = image

    def __call__(self, data):
        d = dict(data)
        img = d[self.image]
        img = np.moveaxis(img, 0, -1)  # to channel last
        img = img * 128 + 128
        img = img.astype(np.uint8)

        for key in self.keys:
            label = d[key].astype(np.uint8)
            label = filter_remove_small_objects(label).astype(np.uint8)
            gray = np.dot(img, [0.2125, 0.7154, 0.0721])
            d[key] = label * np.logical_xor(label, gray == 0)
        return d


class FindContoursd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        min_positive=500,
        min_poly_size=50,
        result="result",
        bbox="bbox",
        contours="contours",
    ):
        super().__init__(keys)

        self.min_positive = min_positive
        self.min_poly_size = min_poly_size
        self.result = result
        self.bbox = bbox
        self.contours = contours

    def __call__(self, data):
        d = dict(data)
        wsi_meta = d.get("wsi", {})
        location = wsi_meta["location"]
        size = wsi_meta["size"]

        tx, ty = location[0], location[1]
        tw, th = size[0], size[1]

        for key in self.keys:
            p = d[key]
            if np.count_nonzero(p) < self.min_positive:
                continue

            bbox = [[tx, ty], [tx + tw, ty + th]]
            contours, _ = cv2.findContours(p, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for cidx, contour in enumerate(contours):
                contour = np.squeeze(contour)
                if len(contour) < self.min_poly_size:  # Ignore poly with less than 3 points
                    continue

                contour[:, 0] += tx  # X
                contour[:, 1] += ty  # Y
                polygons.append(contour.astype(int).tolist())

            if len(polygons):
                if d.get(self.result) is None:
                    d[self.result] = dict()
                d[self.result].update({self.bbox: bbox, self.contours: polygons})
        return d


class AddInitialSeedPointExd(AddInitialSeedPointd):
    def _apply(self, label, sid):
        try:
            return super()._apply(label, sid)
        except AssertionError:
            dimensions = 2
            default_guidance = [-1] * (dimensions + 1)
            return np.asarray([[default_guidance], [default_guidance]])
