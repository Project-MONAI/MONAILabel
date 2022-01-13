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
from typing import Dict, Hashable, Mapping

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

logger = logging.getLogger(__name__)


class SelectLabelsSpineDatasetd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_names=None,
        allow_missing_keys: bool = False,
    ):
        """
        Select labels from list on the VerSe dataset

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        :param label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names
        self.all_label_values = {
            "C1 Atlas": 1,
            "C2 Axis": 2,
            "C3": 3,
            "C4": 4,
            "C5": 5,
            "C6": 6,
            "C7": 7,
            "Th1": 8,
            "Th2": 9,
            "Th3": 10,
            "Th4": 11,
            "Th5": 12,
            "Th6": 13,
            "Th7": 14,
            "Th8": 15,
            "Th9": 16,
            "Th10": 17,
            "Th11": 18,
            "Th12": 19,
            "L1": 20,
            "L2": 21,
            "L3": 22,
            "L4": 23,
            "L5": 24,
            "Sacrum": 25,
        }

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        label_info = d.get("meta", {}).get("label", {}).get("label_info", [])
        remap = {l["name"]: l["idx"] for l in label_info}
        for key in self.key_iterator(d):
            if key == "label":
                new_label_names = dict()
                label = np.zeros(d[key].shape)

                # Making sure the range values and number of labels are the same
                for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                    if key_label != "background":
                        new_label_names[key_label] = idx
                        label[d[key] == remap.get(key_label, val_label)] = idx
                    if key_label == "background":
                        new_label_names["background"] = 0

                d["label_names"] = new_label_names
                d[key] = label
            else:
                print("This transform only applies to the label")
        return d
