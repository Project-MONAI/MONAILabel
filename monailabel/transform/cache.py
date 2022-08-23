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

import copy
import hashlib
import logging
import os
import pathlib
from typing import Hashable, Sequence, Tuple, Union

import torch
from expiring_dict import ExpiringDict
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import Transform
from monai.utils import ensure_tuple

from monailabel.utils.sessions import Sessions

logger = logging.getLogger(__name__)

_cache_path = None
_data_mem_cache = None
_data_file_cache = None


def init_cache():
    global _cache_path
    global _data_mem_cache
    global _data_file_cache
    if not _cache_path:
        _cache_path = os.path.join(pathlib.Path.home(), ".cache", "monailabel", "cacheT")
        _data_mem_cache = ExpiringDict(ttl=600)
        _data_file_cache = Sessions(store_path=_cache_path, expiry=600)

    _data_file_cache.remove_expired()


class CacheTransformDatad(Transform):
    def __init__(
        self,
        keys: KeysCollection,
        hash_key: Union[str, Sequence[str]] = ("image_path", "model"),
        in_memory: bool = True,
        ttl: int = 600,
        reset_applied_operations_id: bool = True,
    ):
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        self.hash_key = [hash_key] if isinstance(hash_key, str) else hash_key
        self.in_memory = in_memory
        self.ttl = ttl
        self.reset_applied_operations_id = reset_applied_operations_id

        # remove previous expired...
        init_cache()

    def __call__(self, data):
        return self.save(data)

    def load(self, data):
        d = dict(data)
        hash_key_prefix = hashlib.md5("".join([d[k] for k in self.hash_key]).encode("utf-8")).hexdigest()

        # full dictionary
        if not self.keys:
            return self._load(f"{hash_key_prefix}")

        # set of keys
        for key in self.keys:
            d[key] = self._load(f"{hash_key_prefix}_{key}")
            if d[key] is None:
                logger.info(f"Ignore; Failed to load {key} from Cache; memory:{self.in_memory}")
                return None

            # For Invert Transform (reset id)
            if self.reset_applied_operations_id and isinstance(d[key], MetaTensor):
                for o in d[key].applied_operations:
                    o["id"] = "none"
        return d

    def save(self, data):
        d = dict(data)

        hash_keys = [d[k] for k in self.hash_key if d.get(k)]
        hash_key_prefix = hashlib.md5("".join(hash_keys).encode("utf-8")).hexdigest()
        if len(hash_keys) != len(self.hash_key):
            logger.warning(f"Ignore caching; Missing hash keys;  Found: {hash_keys}; Expected: {self.hash_key}")
            return d

        # full dictionary
        if not self.keys:
            self._save(f"{hash_key_prefix}", d)
        else:
            for key in self.keys:
                self._save(f"{hash_key_prefix}_{key}", d[key])
        return d

    def _load(self, hash_key):
        if self.in_memory:
            return _data_mem_cache.get(hash_key)

        info = _data_file_cache.get_session(session_id=hash_key)
        if info and os.path.isfile(info.image):
            return torch.load(info.image)
        return None

    def _save(self, hash_key, obj):
        if self.in_memory:
            _data_mem_cache.ttl(key=hash_key, value=copy.deepcopy(obj), ttl=self.ttl)
        else:
            os.makedirs(_cache_path, exist_ok=True)

            cached_file = os.path.join(_cache_path, f"{hash_key}.tmp")
            torch.save(obj, cached_file)
            _data_file_cache.add_session(cached_file, expiry=self.ttl, session_id=hash_key)
