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

import json
import logging
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Union

import numpy as np
import torch
from monai.config import IndexSelection, KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, Randomizable, Resize, SpatialCrop, generate_spatial_bounding_box, is_positive
from monai.utils import InterpolateMode, PostFix, ensure_tuple_rep
from scipy.ndimage import distance_transform_cdt, gaussian_filter
from skimage import measure

logger = logging.getLogger(__name__)


class AddClickGuidanced(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, guidance="guidance"):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance

    def __call__(self, data):
        d = dict(data)
        guidance = []
        for key in self.keys:
            g = d.get(key)
            g = np.array(g).astype(int).tolist() if g else []
            guidance.append(g)

        d[self.guidance] = guidance
        return d


class AddInitialSeedPointd(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, label="label", connected_regions=1):
        super().__init__(keys, allow_missing_keys)

        self.label = label
        self.connected_regions = connected_regions

    def _apply(self, label):
        default_guidance = [-1] * len(label.shape)

        if self.connected_regions > 1:
            blobs_labels = measure.label(label, background=0)
            u, count = np.unique(blobs_labels, return_counts=True)
            count_sort_ind = np.argsort(-count)
            connected_regions = u[count_sort_ind].astype(int).tolist()

            connected_regions = [r for r in connected_regions if r]
            connected_regions = connected_regions[: self.connected_regions]
        else:
            blobs_labels = None
            connected_regions = [1]

        pos_guidance = []
        for region in connected_regions:
            label = label if blobs_labels is None else (blobs_labels == region).astype(int)
            if np.sum(label) == 0:
                continue

            distance = distance_transform_cdt(label).flatten()
            probability = np.exp(distance) - 1.0

            idx = np.where(label.flatten() > 0)[0]
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
            g[0] = dst[0]  # for debug
            pos_guidance.append(g)

        return np.asarray([pos_guidance, [default_guidance] * len(pos_guidance)]).astype(int, copy=False).tolist()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = json.dumps(self._apply(d[self.label]))
        return d


class AddGuidanceSignald(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        guidance: str = "guidance",
        sigma: int = 2,
        number_intensity_ch=3,
    ):
        super().__init__(keys, allow_missing_keys)

        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch

    def signal(self, shape, points):
        signal = np.zeros(shape, dtype=np.float32)
        flag = False
        for p in points:
            if np.any(np.asarray(p) < 0):
                continue
            if len(shape) == 3:
                signal[int(p[-3]), int(p[-2]), int(p[-1])] = 1.0
            else:
                signal[int(p[-2]), int(p[-1])] = 1.0
            flag = True

        if flag:
            signal = gaussian_filter(signal, sigma=self.sigma)
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        return torch.Tensor(signal)[None]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            guidance = d[self.guidance]
            guidance = json.loads(guidance) if isinstance(guidance, str) else guidance
            if guidance and (guidance[0] or guidance[1]):
                img = img[0 : 0 + self.number_intensity_ch, ...]

                shape = img.shape[-2:] if len(img.shape) == 3 else img.shape[-3:]
                device = img.device if isinstance(img, torch.Tensor) else None
                pos = self.signal(shape, guidance[0]).to(device=device)
                neg = self.signal(shape, guidance[1]).to(device=device)
                result = torch.concat([img if isinstance(img, torch.Tensor) else torch.Tensor(img), pos, neg])
            else:
                s = torch.zeros_like(img[0])[None]
                result = torch.concat([img, s, s])

            d[key] = result
        return d


class SpatialCropForegroundd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        spatial_size: Union[Sequence[int], np.ndarray],
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: int = 0,
        allow_smaller: bool = True,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.source_key = source_key
        self.spatial_size = list(spatial_size)
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.allow_smaller = allow_smaller

        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(
            d[self.source_key], self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )

        center = list(np.mean([box_start, box_end], axis=0).astype(int, copy=False))
        current_size = list(np.subtract(box_end, box_start).astype(int, copy=False))

        if np.all(np.less(current_size, self.spatial_size)):
            cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
            box_start = np.array([s.start for s in cropper.slices])
            box_end = np.array([s.stop for s in cropper.slices])
        else:
            cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)

        for key in self.keys:
            image = d[key]
            meta = image.meta
            meta[self.start_coord_key] = box_start
            meta[self.end_coord_key] = box_end
            meta[self.original_shape_key] = d[key].shape

            result = cropper(image)
            meta[self.cropped_shape_key] = result.shape
            d[key] = result
        return d


class RestoreLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        mode: Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str] = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = PostFix.meta(),
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ref_image = ref_image
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))

        self.meta_key_postfix = meta_key_postfix
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        meta_dict = (
            d[self.ref_image].meta
            if isinstance(d[self.ref_image], MetaTensor)
            else d[f"{self.ref_image}_{self.meta_key_postfix}"]
        )

        for key, mode, align_corners in self.key_iterator(d, self.mode, self.align_corners):
            image = d[key]

            # Undo Resize
            current_shape = image.shape
            cropped_shape = meta_dict[self.cropped_shape_key]
            if np.any(np.not_equal(current_shape, cropped_shape)):
                resizer = Resize(spatial_size=cropped_shape[1:], mode=mode)
                image = resizer(image, mode=mode, align_corners=align_corners)

            # Undo Crop
            original_shape = meta_dict[self.original_shape_key][1:]
            result = np.zeros(original_shape, dtype=np.float32)
            box_start = meta_dict[self.start_coord_key]
            box_end = meta_dict[self.end_coord_key]

            spatial_dims = min(len(box_start), len(image.shape[1:]))
            slices = [slice(s, e) for s, e in zip(box_start[:spatial_dims], box_end[:spatial_dims])]
            slices = tuple(slices)
            result[slices] = image.array if isinstance(image, MetaTensor) else image

            d[key] = result
        return d


class SpatialCropGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        guidance: str,
        spatial_size,
        margin=20,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        original_shape_key: str = "foreground_original_shape",
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.guidance = guidance
        self.spatial_size = list(spatial_size)
        self.margin = margin

        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.original_shape_key = original_shape_key
        self.cropped_shape_key = cropped_shape_key

    def bounding_box(self, points, img_shape):
        ndim = len(img_shape)
        margin = ensure_tuple_rep(self.margin, ndim)
        for m in margin:
            if m < 0:
                raise ValueError("margin value should not be negative number.")

        box_start = [0] * ndim
        box_end = [0] * ndim

        for di in range(ndim):
            dt = points[..., di]
            min_d = max(min(dt - margin[di]), 0)
            max_d = min(img_shape[di], max(dt + margin[di] + 1))
            box_start[di], box_end[di] = min_d, max_d
        return box_start, box_end

    def __call__(self, data):
        d: Dict = dict(data)
        first_key: Union[Hashable, List] = self.first_key(d)
        if not first_key:
            return d

        guidance = d[self.guidance]
        original_spatial_shape = d[first_key].shape[1:]
        box_start, box_end = self.bounding_box(np.array(guidance[0] + guidance[1]), original_spatial_shape)
        center = list(np.mean([box_start, box_end], axis=0).astype(int, copy=False))
        spatial_size = self.spatial_size

        box_size = list(np.subtract(box_end, box_start).astype(int, copy=False))
        spatial_size = spatial_size[-len(box_size) :]

        if np.all(np.less(box_size, spatial_size)):
            cropper = SpatialCrop(roi_center=center, roi_size=spatial_size)
        else:
            cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)

        # update bounding box in case it was corrected by the SpatialCrop constructor
        box_start = np.array([s.start for s in cropper.slices])
        box_end = np.array([s.stop for s in cropper.slices])

        for key in self.keys:
            image = d[key]
            meta = image.meta
            meta[self.start_coord_key] = box_start
            meta[self.end_coord_key] = box_end
            meta[self.original_shape_key] = d[key].shape

            result = cropper(image)
            result.meta[self.cropped_shape_key] = result.shape
            d[key] = result

        pos_clicks, neg_clicks = guidance[0], guidance[1]
        pos = np.subtract(pos_clicks, box_start).tolist() if len(pos_clicks) else []
        neg = np.subtract(neg_clicks, box_start).tolist() if len(neg_clicks) else []

        d[self.guidance] = [pos, neg]
        return d


class ResizeGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        cropped_shape_key: str = "foreground_cropped_shape",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ref_image = ref_image
        self.cropped_shape_key = cropped_shape_key

    def __call__(self, data):
        d = dict(data)
        current_shape = d[self.ref_image].shape[1:]

        meta = d[self.ref_image].meta
        if self.cropped_shape_key and meta.get(self.cropped_shape_key):
            cropped_shape = meta[self.cropped_shape_key][1:]
        else:
            cropped_shape = meta.get("spatial_shape", current_shape)
        factor = np.divide(current_shape, cropped_shape)

        for key in self.keys:
            guidance = d[key]
            pos_clicks, neg_clicks = guidance[0], guidance[1]
            pos = np.multiply(pos_clicks, factor).astype(int, copy=False).tolist() if len(pos_clicks) else []
            neg = np.multiply(neg_clicks, factor).astype(int, copy=False).tolist() if len(neg_clicks) else []

            d[key] = [pos, neg]
        return d
