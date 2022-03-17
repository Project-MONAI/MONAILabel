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
import logging
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms import MapTransform, Randomizable, RandSpatialCropSamplesd, SpatialCrop, Transform
from monai.utils import ImageMetaKey, ensure_tuple, ensure_tuple_rep

logger = logging.getLogger(__name__)


class FindAllValidSlicesByClassd(Transform):
    """
    Find/List all valid slices in the label for each class.
    Label is assumed to be a 3D Volume(DHW) or 4D Volume with shape CDHW, where C=1.

    Args:
        label: key to the label source.
        sids: key to store slices indices having valid label map.
    """

    def __init__(self, label: str = "label", sids: str = "sids"):
        self.label = label
        self.sids = sids

    def _apply(self, label):
        labels_idx = [int(idx) for idx in np.unique(label) if idx]  # exclude background

        res = {}
        for label_idx in labels_idx:
            sids = []
            for sid in range(label.shape[2]):
                if np.any(label[:, :, sid] == label_idx):
                    sids.append(sid)
            res[label_idx] = sids
        return res

    def __call__(self, data):
        d: Dict = dict(data)
        label = d[self.label]
        if len(label.shape) not in (3, 4):
            raise ValueError("Only supports label with shape CDHW or DHW")
        if len(label.shape) == 4 and label.shape[0] != 1:
            raise ValueError("Only supports single channel labels for CDHW!")

        d[self.sids] = self._apply(label if len(label.shape) == 3 else label[0])
        logger.debug("sids for {} => {}".format(label.shape, {k: len(v) for k, v in d[self.sids].items()}))
        return d


class RandomForegroundCropSamplesd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        sids_key: str = "sids",
        num_samples: int = 1,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.label_key = label_key
        self.sids_key = sids_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        self.num_samples = num_samples

        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        self.meta_key_postfix = meta_key_postfix

    def randomize(self, data):
        pass

    def _apply(self, label, sids):
        idx = self.R.choice([int(k) for k in sids.keys()], replace=False)
        sid = self.R.choice(sids[idx], replace=False)

        label_2d = label[0][:, :, sid] if len(label.shape) == 4 else label[:, :, sid]
        x, y = np.where(np.equal(label_2d, idx))
        box_start = x.min(), y.min()
        box_end = x.max(), y.max()
        center = list(np.mean([box_start, box_end], axis=0).astype(int, copy=False))

        # print(f"Selected {idx} => {sid} => {center} => {label.shape}")
        return [center[0], center[1], sid]

    def __call__(self, data):
        d = dict(data)

        label = d[self.label_key]
        sids = d[self.sids_key]

        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(self.num_samples)]
        for i in range(self.num_samples):
            center = self._apply(label, sids)
            cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)

            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = copy.deepcopy(d[key])

            for key in self.key_iterator(d):
                results[i][key] = cropper(d[key])

            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][ImageMetaKey.PATCH_INDEX] = i  # type: ignore

        return results


class RandomCroppedSamplesd(Randomizable, Transform):
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        sids_key: str = "sids",
        num_samples: int = 1,
        foreground_probability: float = 0.9,
    ):
        self.probability = foreground_probability
        self.random_foreground = RandomForegroundCropSamplesd(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            sids_key=sids_key,
            num_samples=num_samples,
        )
        self.random_spatial = RandSpatialCropSamplesd(
            keys=keys,
            roi_size=spatial_size,
            max_roi_size=spatial_size,
            random_center=True,
            random_size=False,
            num_samples=num_samples,
        )

    def randomize(self, data):
        pass

    def __call__(self, data):
        if self.R.choice([True, False], p=[self.probability, 1 - self.probability]):
            return self.random_foreground(data)
        return self.random_spatial(data)
