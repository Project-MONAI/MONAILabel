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
import os.path
from typing import Dict, Optional

import numpy as np
import torch
from monai.transforms import LoadImage

from monailabel.datastore.cvat import CVATDatastore
from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.bundle import BundleConstants, BundleTrainTask

logger = logging.getLogger(__name__)


class InBody(BundleTrainTask):
    def __init__(self, path: str, conf: Dict[str, str], const: Optional[BundleConstants] = None):
        super().__init__(path, conf, const, enable_tracking=True)

    def _fetch_datalist(self, request, datastore: Datastore):
        ds = super()._fetch_datalist(request, datastore)

        out_body = datastore.label_map.get("OutBody", 3) if isinstance(datastore, CVATDatastore) else 1
        load = LoadImage(dtype=np.uint8, image_only=True)

        for d in ds:
            label = d.get("label")
            if label is not None and isinstance(label, str) and os.path.exists(label):
                d["label"] = int(torch.max(torch.where(load(d["label"]) == out_body, 1, 0)))
        return ds
