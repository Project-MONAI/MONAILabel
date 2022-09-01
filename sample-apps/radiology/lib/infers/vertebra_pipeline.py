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
import time
from typing import Callable, Sequence

import numpy as np
from tqdm import tqdm

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.interfaces.utils.transform import run_transforms
from monailabel.transform.post import Restored
from monailabel.transform.writer import Writer

logger = logging.getLogger(__name__)


class InferVertebraPipeline(InferTask):
    def __init__(
        self,
        task_loc_spine: InferTask,
        task_loc_vertebra: InferTask,
        task_seg_vertebra: InferTask,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="Combines three stages for vertebra segmentation",
        **kwargs,
    ):
        super().__init__(
            path=None,
            network=None,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.task_loc_spine = task_loc_spine
        self.task_loc_vertebra = task_loc_vertebra
        self.task_seg_vertebra = task_seg_vertebra

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def _latencies(self, r, e=None):
        if not e:
            e = {"pre": 0, "infer": 0, "invert": 0, "post": 0, "write": 0, "total": 0}

        for key in e:
            e[key] = e[key] + r.get("latencies", {}).get(key, 0)
        return e

    def locate_spine(self, request):
        req = copy.deepcopy(request)
        req.update({"pipeline_mode": True})

        d, r = self.task_loc_spine(req)
        return d, r, self._latencies(r)

    def locate_vertebra(self, request, image, label):
        req = copy.deepcopy(request)
        req.update({"image": image, "label": label, "pipeline_mode": True})

        d, r = self.task_loc_vertebra(req)
        return d, r, self._latencies(r)

    def segment_vertebra(self, request, image, centroids):
        current_size = list(image.shape)
        result_mask = np.zeros(current_size, np.float)

        l = None
        for centroid in tqdm(centroids):
            req = copy.deepcopy(request)
            req.update(
                {
                    "image": image,
                    "original_size": current_size,
                    "centroids": [centroid],
                    "pipeline_mode": True,
                    "logging": "ERROR" if l else "INFO",
                }
            )

            d, r = self.task_seg_vertebra(req)
            l = self._latencies(r, l)

            # Paste each mask
            c = d["slices_cropped"]
            s = d["cropped_size"]
            c00 = c[0][0]
            c01 = c00 + s[0]
            c10 = c[1][0]
            c11 = c10 + s[1]
            c20 = c[2][0]
            c21 = c20 + s[2]

            m = d["pred"].array
            m = m[:, : s[0], : s[1], : s[2]]
            result_mask[:, c00:c01, c10:c11, c20:c21] = m

        return result_mask, l

    def __call__(self, request):
        start = time.time()
        request.update({"image_path": request.get("image")})

        # Run first stage
        d1, r1, l1 = self.locate_spine(request)
        image = d1["image"]
        label = d1["pred"]

        # Run second stage
        d2, r2, l2 = self.locate_vertebra(request, image, label)
        centroids = r2["centroids"]

        # Run third stage
        result_mask, l3 = self.segment_vertebra(request, image, centroids)

        # Finalize the mask/result
        data = copy.deepcopy(request)
        data.update({"pred": result_mask, "image": image})
        data = run_transforms(data, [Restored(keys="pred", ref_image="image")], log_prefix="POST", use_compose=False)

        begin = time.time()
        result_file, _ = Writer(label="pred")(data)
        latency_write = round(time.time() - begin, 2)

        total_latency = round(time.time() - start, 2)
        result_json = {
            "labels": self.labels,
            "centroids": centroids,
            "latencies": {
                "locate_spine": l1,
                "locate_vertebra": l2,
                "segment_vertebra": l3,
                "write": latency_write,
                "total": total_latency,
            },
        }

        logger.info(f"Result Mask (aggregated): {result_mask.shape}; total_latency: {total_latency}")
        return result_file, result_json
