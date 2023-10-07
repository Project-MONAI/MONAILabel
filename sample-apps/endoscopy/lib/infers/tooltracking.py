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
from typing import Any, Callable, Dict, Sequence

from monai.transforms import SqueezeDimd

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class ToolTracking(BundleInferTask):
    """
    This provides Inference Engine for pre-trained segmentation model for Tool Tracking.
    """

    def __init__(self, path: str, conf: Dict[str, str], **kwargs):
        super().__init__(path, conf, type=InferType.SEGMENTATION, load_strict=False, **kwargs)

        # Override Labels
        self.labels = {"Tool": 1}
        self.label_colors = {"Tool": (255, 0, 0)}

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["endoscopy"] = True
        return d

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = list(super().post_transforms())
        t.append(SqueezeDimd(keys="pred", dim=0))
        t.append(FindContoursd(keys="pred", labels=self.labels))
        return t

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
