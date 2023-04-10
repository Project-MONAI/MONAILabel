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
import logging
import os
import pathlib
from typing import Dict, Optional

import cv2
import numpy as np
from lib.hovernet import PatchExtractor
from lib.utils import split_dataset
from PIL import Image
from tqdm import tqdm

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.bundle import BundleConstants, BundleTrainTask
from monailabel.utils.others.generic import remove_file

logger = logging.getLogger(__name__)


class HovernetNuclei(BundleTrainTask):
    def __init__(self, path: str, conf: Dict[str, str], const: Optional[BundleConstants] = None):
        super().__init__(path, conf, const, enable_tracking=True)
        self.tile_size = (1024, 1024)
        self.patch_size = (540, 540)
        self.step_size = (164, 164)
        self.extract_type = "mirror"

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
            groups=None,
            tile_size=self.tile_size,
            max_region=max_region,
            limit=request.get("dataset_limit", 0),
            randomize=request.get("dataset_randomize", True),
        )
        logger.info(f"Split data (len: {len(ds)}) based on each nuclei")

        limit = request.get("dataset_limit", 0)
        ds_new: list = []
        xtractor = PatchExtractor(self.patch_size, self.step_size)
        out_dir = os.path.join(cache_dir, "nuclei_hovernet")
        os.makedirs(out_dir, exist_ok=True)

        for d in tqdm(ds):
            if 0 < limit < len(ds_new):
                ds_new = ds_new[:limit]
                break

            base_name = pathlib.Path(d["image"]).stem
            img = np.array(Image.open(d["image"]).convert("RGB"))
            ann_type = np.array(Image.open(d["label"]))

            numLabels, ann_inst, _, _ = cv2.connectedComponentsWithStats(ann_type, 4, cv2.CV_32S)
            ann = np.dstack([ann_inst, ann_type])

            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, self.extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm(total=len(sub_patches), leave=False, bar_format=pbar_format, ascii=True, position=1)

            for idx, patch in enumerate(sub_patches):
                image_patch = patch[..., :3]
                inst_map_patch = patch[..., 3:4]
                type_map_patch = patch[..., 4:5]

                i = f"{out_dir}/{base_name}_{idx:03d}_image.npy"
                j = f"{out_dir}/{base_name}_{idx:03d}_inst_map.npy"
                k = f"{out_dir}/{base_name}_{idx:03d}_type_map.npy"

                np.save(i, image_patch)
                np.save(j, inst_map_patch)
                np.save(k, type_map_patch)
                ds_new.append({"image": i, "label_inst": j, "label_type": k})
                pbar.update()
            pbar.close()

            if 0 < limit < len(ds_new):
                ds_new = ds_new[:limit]
                break

        logger.info(f"Final Records with hovernet patches: {len(ds_new)}")
        return ds_new

    def _load_checkpoint(self, output_dir, pretrained, train_handlers):
        pass

    def run_single_gpu(self, request, overrides):
        logger.info("+++++++++++ Running STAGE 0.........................")
        overrides["stage"] = 0
        overrides["network_def#freeze_encoder"] = True
        pretrained = os.path.join(self.bundle_path, "models", "stage0", "model.pt")
        if os.path.exists(pretrained):
            overrides["network_def#pretrained_url"] = pathlib.Path(pretrained).as_uri()
        super().run_single_gpu(request, overrides)

        logger.info("+++++++++++ Running STAGE 1.........................")
        overrides["stage"] = 1
        overrides["network_def#freeze_encoder"] = False
        overrides["network_def#pretrained_url"] = None
        super().run_single_gpu(request, overrides)

    def run_multi_gpu(self, request, cmd, env):
        logger.info("+++++++++++ Running STAGE 0.........................")
        cmd1 = copy.deepcopy(cmd)
        cmd1.extend(["--stage", "0", "--network_def#freeze_encoder", "true"])
        pretrained = os.path.join(self.bundle_path, "models", "stage0", "model.pt")
        if os.path.exists(pretrained):
            cmd1.extend(["--network_def#pretrained_url", pathlib.Path(pretrained).as_uri()])
        super().run_multi_gpu(request, cmd1, env)

        logger.info("+++++++++++ Running STAGE 1.........................")
        cmd2 = copy.deepcopy(cmd)
        cmd2.extend(["--stage", "1", "--network_def#freeze_encoder", "false"])
        cmd2.extend(["--network_def#pretrained_url", "None"])
        super().run_multi_gpu(request, cmd2, env)

    def __call__(self, request, datastore: Datastore):
        request["force_multi_gpu"] = True
        return super().__call__(request, datastore)
