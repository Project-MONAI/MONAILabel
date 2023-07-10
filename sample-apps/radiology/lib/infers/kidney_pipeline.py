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

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask

logger = logging.getLogger(__name__)


class InferKidneyPipeline(BasicInferTask):
    def __init__(
        self,
        task_seg: InferTask,
        task_seg_tumor: InferTask,
        type=InferType.SEGMENTATION,
        description="Combines two stages for kidney and kidney tumor segmentation",
        **kwargs,
    ):
        self.task_seg = task_seg
        self.task_seg_tumor = task_seg_tumor

        super().__init__(
            path=None,
            network=None,
            type=type,
            labels=task_seg_tumor.labels,
            dimension=task_seg_tumor.dimension,
            description=description,
            **kwargs,
        )

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def is_valid(self) -> bool:
        return True

    def _latencies(self, r, e=None):
        if not e:
            e = {"pre": 0, "infer": 0, "invert": 0, "post": 0, "write": 0, "total": 0}

        for key in e:
            e[key] = e[key] + r.get("latencies", {}).get(key, 0)
        return e

    def segment_kidney(self, request):
        req = copy.deepcopy(request)
        req.update({"pipeline_mode": True})

        d, r = self.task_seg(req)
        return d, r, self._latencies(r)

    def segment_tumor(self, request, image, label):
        req = copy.deepcopy(request)
        req.update({"image": image, "label": label, "pipeline_mode": True})

        d, r = self.task_seg_tumor(req)
        return d, self._latencies(r)

    def __call__(self, request):
        start = time.time()
        request.update({"image_path": request.get("image")})

        # Run first stage
        d1, r1, l1 = self.segment_kidney(request)
        image = d1["image"]
        label = d1["pred"]

        # Run second stage
        result_mask, l2 = self.segment_tumor(request, image, label)

        total_latency = round(time.time() - start, 2)
        result_json = {
            "label_names": self.task_seg_tumor.labels,
            "latencies": {
                "segmentation": l1,
                "kidney_tumor": l2,
                "total": total_latency,
            },
        }

        logger.info(f"total_latency: {total_latency}")
        return result_mask, result_json
