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
import os
from typing import Dict, Optional

from lib.utils import split_dataset, split_nuclei_dataset
from tqdm import tqdm

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.bundle import BundleConstants, BundleTrainTask
from monailabel.utils.others.generic import remove_file

logger = logging.getLogger(__name__)


class ClassificationNuclei(BundleTrainTask):
    def __init__(self, path: str, conf: Dict[str, str], const: Optional[BundleConstants] = None):
        super().__init__(path, conf, const, enable_tracking=True)
        self.labels = {
            "Other": 1,
            "Inflammatory": 2,
            "Epithelial": 3,
            "Spindle-Shaped": 4,
        }
        self.tile_size = (256, 256)

    def _fetch_datalist(self, request, datastore: Datastore):
        cache_dir = os.path.join(self.bundle_path, "cache", "train_ds")
        remove_file(cache_dir)

        source = request.get("dataset_source")
        max_region = request.get("dataset_max_region", (10240, 10240))
        max_region = (max_region, max_region) if isinstance(max_region, int) else max_region[:2]

        ds = split_dataset(
            datastore=datastore,
            cache_dir=cache_dir,
            source=source,
            groups=self.labels,
            tile_size=self.tile_size,
            max_region=max_region,
            limit=request.get("dataset_limit", 0),
            randomize=request.get("dataset_randomize", True),
        )
        logger.info(f"Split data (len: {len(ds)}) based on each nuclei")

        limit = request.get("dataset_limit", 0)
        ds_new = []
        for d in tqdm(ds):
            ds_new.extend(split_nuclei_dataset(d, os.path.join(cache_dir, "nuclei_flattened")))
            if 0 < limit < len(ds_new):
                ds_new = ds_new[:limit]
                break
        logger.info(f"Final Records with nuclei split: {len(ds_new)}")
        return ds_new

    def _update_overrides(self, overrides):
        overrides = super()._update_overrides(overrides)

        overrides["train_datalist"] = ""
        overrides["val_datalist"] = ""
        return overrides
