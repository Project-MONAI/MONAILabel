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
import os
import time
from functools import partial

import numpy as np
import torch
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandAffined,
    RandFlipd,
    RandRotated,
    Resized,
    Spacingd,
    ToTensord,
)

from monailabel.deepedit.transforms import DiscardAddGuidanced, SingleLabelSingleModalityd
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.tasks.infer.tta import TestTimeAugmentation

logger = logging.getLogger(__name__)


class TTAScoring(ScoringMethod):
    """
    First version of test time augmentation active learning
    """

    def __init__(self, model, network=None, deepedit=True, num_samples=5):
        super().__init__("Compute initial score based on TTA")
        self.model = model
        self.device = "cuda"
        self.img_size = [128, 128, 128]
        self.num_samples = num_samples
        self.network = network
        self.deepedit = deepedit

    def pre_transforms(self):
        t = [
            LoadImaged(keys="image", reader="nibabelreader"),
            SingleLabelSingleModalityd(keys="image"),
            AddChanneld(keys="image"),
            Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0]),
            RandAffined(
                keys="image",
                prob=1,
                rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                padding_mode="zeros",
                as_tensor_output=False,
            ),
            RandFlipd(keys="image", prob=0.5, spatial_axis=0),
            RandRotated(keys="image", range_x=(-5, 5), range_y=(-5, 5), range_z=(-5, 5)),
            Resized(keys="image", spatial_size=self.img_size),
        ]
        # If using TTA for deepedit
        if self.deepedit:
            t.append(DiscardAddGuidanced(keys="image"))
        t.append(ToTensord(keys="image"))
        return Compose(t)

    def post_transforms(self):
        return Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold_values=True),
            ]
        )

    def _inferer(self, images, model):
        preds = SimpleInferer()(images, model)
        transforms = self.post_transforms()
        post_pred = transforms(preds)
        return post_pred

    def _get_model_path(self, path):
        if not path:
            return None

        paths = [path] if isinstance(path, str) else path
        for path in reversed(paths):
            if os.path.exists(path):
                return path
        return None

    def __call__(self, request, datastore: Datastore):
        logger.info("Starting TTA scoring")

        result = {}
        model_file = self._get_model_path(self.model)
        if not model_file:
            logger.warning(f"Skip TTA Scoring:: Model(s) {self.model} not available yet")
            return

        logger.info(f"Using {model_file} for running TTA")
        model_ts = int(os.stat(model_file).st_mtime)
        if self.network:
            model = self.network
            if model_file:
                checkpoint = torch.load(model_file)
                model_state_dict = checkpoint.get("model", checkpoint)
                model.load_state_dict(model_state_dict)
        else:
            model = torch.jit.load(model_file)

        tt_aug = TestTimeAugmentation(
            transform=self.pre_transforms(),
            label_key="image",
            batch_size=1,
            num_workers=0,
            inferrer_fn=partial(self._inferer, model=model.to(self.device)),
            device=self.device,
            progress=self.num_samples > 1,
        )

        # Performing TTA for all unlabeled images
        skipped = 0
        unlabeled_images = datastore.get_unlabeled_images()
        num_samples = request.get("num_samples", self.num_samples)

        for image_id in unlabeled_images:
            image_info = datastore.get_image_info(image_id)
            prev_ts = image_info.get("tta_ts", 0)
            if prev_ts == model_ts:
                skipped += 1
                continue

            logger.info(f"TTA:: Run for image: {image_id}; Prev Ts: {prev_ts}; New Ts: {model_ts}")

            # Computing the Volume Variation Coefficient (VVC)
            start = time.time()
            with torch.no_grad():
                data = {"image": datastore.get_image_uri(image_id)}
                mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(data, num_examples=num_samples)

            logger.info(f"TTA:: {image_id} => vvc: {vvc_tta}")
            if self.device == "cuda":
                torch.cuda.empty_cache()

            latency_tta = time.time() - start
            logger.info(f"TTA:: Time taken for {num_samples} augmented samples: {latency_tta} (sec)")

            # Add vvc in datastore
            info = {"vvc_tta": vvc_tta, "tta_ts": model_ts}
            datastore.update_image_info(image_id, info)
            result[image_id] = info

        logger.info(f"TTA:: Total: {len(unlabeled_images)}; Skipped = {skipped}; Executed: {len(result)}")
        return result
