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

from typing import Optional, Sequence, Union

import numpy as np
import skimage.measure as measure
from monai.config import KeysCollection
from monai.transforms import MapTransform, Resize, generate_spatial_bounding_box, get_extreme_points
from monai.transforms.spatial.dictionary import InterpolateModeSequence
from monai.utils import InterpolateMode, ensure_tuple_rep

# TODO:: Move to MONAI ??


class LargestCCd(MapTransform):
    def __init__(self, keys: KeysCollection, has_channel: bool = True):
        super().__init__(keys)
        self.has_channel = has_channel

    @staticmethod
    def get_largest_cc(label):
        largest_cc = np.zeros(shape=label.shape, dtype=label.dtype)
        for i, item in enumerate(label):
            item = measure.label(item, connectivity=1)
            if item.max() != 0:
                largest_cc[i, ...] = item == (np.argmax(np.bincount(item.flat)[1:]) + 1)
        return largest_cc

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = self.get_largest_cc(d[key] if self.has_channel else d[key][np.newaxis])
            d[key] = result if self.has_channel else result[0]
        return d


class ExtremePointsd(MapTransform):
    def __init__(self, keys: KeysCollection, result: str = "result", points: str = "points"):
        super().__init__(keys)
        self.result = result
        self.points = points

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            try:
                points = get_extreme_points(d[key])
                if d.get(self.result) is None:
                    d[self.result] = dict()
                d[self.result][self.points] = np.array(points).astype(int).tolist()
            except ValueError:
                pass
        return d


class BoundingBoxd(MapTransform):
    def __init__(self, keys: KeysCollection, result: str = "result", bbox: str = "bbox"):
        super().__init__(keys)
        self.result = result
        self.bbox = bbox

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            bbox = generate_spatial_bounding_box(d[key])
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result][self.bbox] = np.array(bbox).astype(int).tolist()
        return d


class Restored(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        has_channel: bool = True,
        mode: InterpolateModeSequence = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        super().__init__(keys)
        self.ref_image = ref_image
        self.has_channel = has_channel
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        meta_dict = d[f"{self.ref_image}_{self.meta_key_postfix}"]
        for idx, key in enumerate(self.keys):
            result = d[key]
            current_size = result.shape[1:] if self.has_channel else result.shape
            spatial_shape = meta_dict["spatial_shape"]
            spatial_size = spatial_shape[-len(current_size) :]

            # Undo Spacing
            if np.any(np.not_equal(current_size, spatial_size)):
                resizer = Resize(spatial_size=spatial_size, mode=self.mode[idx])
                result = resizer(result, mode=self.mode[idx], align_corners=self.align_corners[idx])

            d[key] = result if len(result.shape) <= 3 else result[0] if result.shape[0] == 1 else result

            meta = d.get(f"{key}_{self.meta_key_postfix}")
            if meta is None:
                meta = dict()
                d[f"{key}_{self.meta_key_postfix}"] = meta
            meta["affine"] = meta_dict.get("original_affine")
        return d
