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

from monai.config import KeysCollection
from monai.handlers import MeanDice, from_engine
from monai.utils import ensure_tuple


def region_wise_metrics(regions, metric, prefix, keys=("pred", "label")):
    all_metrics = dict()
    all_metrics[metric] = MeanDice(output_transform=from_engine(keys), include_background=False)

    if regions:
        labels = regions if isinstance(regions, dict) else {name: idx for idx, name in enumerate(regions, start=1)}
        for name, idx in labels.items():
            all_metrics[f"{prefix}_{name}_dice"] = MeanDice(
                output_transform=from_engine_idx(keys, idx),
                include_background=False,
            )
    return all_metrics


def from_engine_idx(keys: KeysCollection, idx, first: bool = False):
    keys = ensure_tuple(keys)

    def _match(t, idx):
        p, label = t
        p_n = [x[idx, ...][None] for x in p]
        l_n = [x[idx, ...][None] for x in label]
        return p_n, l_n

    def _wrapper(data):
        if isinstance(data, dict):
            return _match(tuple(data[k] for k in keys), idx)

        if isinstance(data, list) and isinstance(data[0], dict):
            ret = [data[0][k] if first else [i[k] for i in data] for k in keys]
            return _match(tuple(ret) if len(ret) > 1 else ret[0], idx)

    return _wrapper
