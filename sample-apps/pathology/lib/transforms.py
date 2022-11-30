# Copyright (c) MONAI Consortium
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
import pathlib

import numpy as np
import openslide
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils import ensure_tuple
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_holes, remove_small_objects

logger = logging.getLogger(__name__)


class LoadImagePatchd(MapTransform):
    def __init__(
        self, keys: KeysCollection, meta_key_postfix: str = "meta_dict", mode="RGB", dtype=np.uint8, padding=True
    ):
        super().__init__(keys)
        self.meta_key_postfix = meta_key_postfix
        self.mode = mode
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
                d["location"] = [0, 0]
                d["size"] = [0, 0]

            img = img.convert(self.mode) if self.mode else img
            image_np = np.array(img, dtype=self.dtype)
            image_np = np.moveaxis(image_np, 0, 1)

            meta_dict_key = f"{key}_{self.meta_key_postfix}"
            meta_dict = d.get(meta_dict_key)
            if meta_dict is None:
                d[meta_dict_key] = dict()
                meta_dict = d.get(meta_dict_key)

            meta_dict["spatial_shape"] = np.asarray(image_np.shape[:-1])
            meta_dict["original_channel_dim"] = -1
            logger.debug(f"Image shape: {image_np.shape} vs size: {size} vs tile_size: {tile_size}")

            if self.padding and tile_size and (image_np.shape[0] != tile_size[0] or image_np.shape[1] != tile_size[1]):
                image_np = self.pad_to_shape(image_np, tile_size)
            d[key] = image_np
        return d

    @staticmethod
    def pad_to_shape(img, shape):
        img_shape = img.shape[:-1]
        s_diff = np.array(shape) - np.array(img_shape)
        diff = [(0, s_diff[0]), (0, s_diff[1]), (0, 0)]
        return np.pad(
            img,
            diff,
            mode="constant",
            constant_values=0,
        )


class PostFilterLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, min_size=64, min_hole=64):
        super().__init__(keys)
        self.min_size = min_size
        self.min_hole = min_hole

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key].astype(np.uint8)
            if self.min_hole:
                label = remove_small_holes(label, area_threshold=self.min_hole)
            label = binary_fill_holes(label).astype(np.uint8)
            if self.min_size:
                label = remove_small_objects(label, min_size=self.min_size)

            d[key] = np.where(label > 0, d[key], 0)
        return d


class ConvertInteractiveClickSignals(MapTransform):
    """
    ConvertInteractiveClickSignals converts interactive annotation information (e.g. from DSA) into a format expected
    by NuClick. Typically, it will take point annotations from data["annotations"][<source_annotation_key>], convert
    it to 2d points, and place it in data[<target_data_key>].
    """

    def __init__(
        self, source_annotation_keys: KeysCollection, target_data_keys: KeysCollection, allow_missing_keys: bool = False
    ):
        super().__init__(target_data_keys, allow_missing_keys)
        self.source_annotation_keys = ensure_tuple(source_annotation_keys)
        self.target_data_keys = ensure_tuple(target_data_keys)

    def __call__(self, data):
        data = dict(data)
        annotations = data.get("annotations", {})
        annotations = {} if annotations is None else annotations
        for source_annotation_key, target_data_key in zip(self.source_annotation_keys, self.target_data_keys):
            if source_annotation_key in annotations:
                points = annotations.get(source_annotation_key)["points"]
                print(f"points={points}")
                points = [coords[0:2] for coords in points]
                data[target_data_key] = points
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"source_annotation_key={source_annotation_key} not found in annotations.keys()={annotations.keys()}"
                )
        return data
