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

import numpy as np
from lib.transforms import LoadImagePatchd, PostFilterLabeld
from monai.transforms import FromMetaTensord, LoadImaged, SaveImaged, SqueezeDimd

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.transform.post import FindContoursd, RenameKeyd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class HovernetNuclei(BundleInferTask):
    """
    This provides Inference Engine for pre-trained Hovernet segmentation + Classification model.
    """

    def __init__(self, path: str, conf: Dict[str, str], **kwargs):
        super().__init__(
            path,
            conf,
            type=InferType.SEGMENTATION,
            add_post_restore=False,
            pre_filter=[LoadImaged],
            post_filter=[FromMetaTensord, SaveImaged],
            load_strict=True,
            **kwargs,
        )

        # Override Labels
        self.labels = {
            "Other": 1,
            "Inflammatory": 2,
            "Epithelial": 3,
            "Spindle-Shaped": 4,
        }
        self.label_colors = {
            "Other": (255, 0, 0),
            "Inflammatory": (255, 255, 0),
            "Epithelial": (0, 0, 255),
            "Spindle-Shaped": (0, 255, 0),
        }
        self._config["label_colors"] = self.label_colors

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        t = [LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False)]
        t.extend([x for x in super().pre_transforms(data)])
        return t

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = [x for x in super().post_transforms(data)]
        t.extend(
            [
                RenameKeyd(source_key="type_map", target_key="pred"),
                SqueezeDimd(keys="pred", dim=0),
                PostFilterLabeld(keys="pred"),
                FindContoursd(keys="pred", labels=self.labels, max_poly_area=128 * 128),
            ]
        )
        return t

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        return d

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
