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
from typing import Optional

from monai.config import KeysCollection
from monai.data import ImageReader, MetaTensor
from monai.transforms import LoadImaged, MapTransform

logger = logging.getLogger(__name__)


class LoadImageExd(LoadImaged):
    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)

        ignore = False
        for i, key in enumerate(self.keys):
            # Support direct image in np (pass only transform)
            if not isinstance(d[key], str):
                ignore = True
                meta_dict_key = f"{key}_{self.meta_key_postfix[i]}"
                meta_dict = d.get(meta_dict_key)
                if meta_dict is None:
                    d[meta_dict_key] = dict()
                    meta_dict = d.get(meta_dict_key)

                image_np = d[key]
                meta_dict["spatial_shape"] = image_np.shape[:-1]  # type: ignore
                meta_dict["original_channel_dim"] = -1  # type: ignore
                meta_dict["original_affine"] = None  # type: ignore

                d[key] = MetaTensor(image_np, meta=meta_dict)
                continue

        if not ignore:
            d = super().__call__(d, reader)

        return d


class NormalizeLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, value=1) -> None:
        super().__init__(keys, allow_missing_keys)
        self.value = value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key].array
            label[label > 0] = self.value
            d[key].array = label
        return d
