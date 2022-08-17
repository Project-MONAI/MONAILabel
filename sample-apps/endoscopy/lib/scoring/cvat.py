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
import random
from typing import Any, Dict

from monailabel.datastore.cvat import CVATDatastore
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.tasks.scoring.epistemic_v2 import EpistemicScoring

logger = logging.getLogger(__name__)


class CVATEpistemicScoring(EpistemicScoring):
    def __init__(
        self,
        top_k,
        infer_task: InferTask,
        num_samples=10,
        use_variance=False,
    ):
        self.top_k = top_k
        super().__init__(infer_task, num_samples, use_variance)

    def get_top_k(self, datastore: Datastore):
        scores: Dict[str, Any] = {}
        for image in datastore.get_unlabeled_images():
            info = datastore.get_image_info(image)
            scores[image] = info.get(self.key_output_entropy, 0)

        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}  # type: ignore

        # shuffle dictionary for similar scores...
        images = list(scores.keys())
        random.shuffle(images)
        scores = {k: {"score": scores[k], "path": datastore.get_image_uri(k)} for k in images}

        top_k: Dict[str, Any] = {}
        max_len = self.top_k if 0 < self.top_k < len(scores) else len(scores)
        for k, v in scores.items():
            if len(top_k) == max_len:
                break
            top_k[k] = v
        return top_k

    def __call__(self, request, datastore: Datastore):
        res = super().__call__(request, datastore)

        if self.top_k and isinstance(datastore, CVATDatastore):
            if datastore.task_status() is None and res and res.get("executed"):
                logger.info(f"Creating new CVAT Task for top-k {self.top_k} samples")
                top_k = self.get_top_k(datastore)
                logger.info(f"{self.key_output_entropy}: Top-N: {list(top_k.keys())}")

                if top_k:
                    images = [top_k[image]["path"] for image in top_k]
                    datastore.upload_to_cvat(images)
            else:
                logger.info("Latest Active Learning Task in CVAT is under progress/not-consumed. Skip to create new!")

        return res
