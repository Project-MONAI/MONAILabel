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
from monai.transforms import MapTransform, Randomizable, SpatialCrop
from monai.utils import ImageMetaKey, ensure_tuple, ensure_tuple_rep

logger = logging.getLogger(__name__)


class RandomCropForegroundd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        num_samples: int = 1,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        self.num_samples = num_samples

        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        self.meta_key_postfix = meta_key_postfix

    def randomize(self, data):
        pass

    def _apply(self, label, label_idx, sids):
        sid = self.R.choice(sids, replace=False)
        label = label[0][sid]

        idx_arr = np.argwhere(label == label_idx)
        random_idx = self.R.randint(len(idx_arr))
        j, k = idx_arr[random_idx]
        return [sid, j, k]

    def __call__(self, data):
        d = dict(data)

        label = d[self.label_key]
        labels_idx = np.unique(label)
        labels_idx = [int(idx) for idx in labels_idx if idx]
        label_sids = {}

        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(self.num_samples)]
        for i in range(self.num_samples):
            label_idx = self.R.choice(labels_idx, replace=False)

            sids = label_sids.get(label_idx, [])
            if not sids:
                for sid in range(label.shape[1]):
                    if np.any(label[0][sid] == label_idx):
                        sids.append(sid)
                label_sids[label_idx] = sids

            center = self._apply(label, label_idx, sids)

            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = copy.deepcopy(d[key])

            for key in self.key_iterator(d):
                img = d[key]
                cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
                results[i][key] = cropper(img)

            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][ImageMetaKey.PATCH_INDEX] = i  # type: ignore

        return results
