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
from typing import Any, Dict

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.bundle import BundleInferTask

logger = logging.getLogger(__name__)


class ClassificationNuclei(BundleInferTask):
    """
    This provides Inference Engine for pre-trained classification model.
    """

    def __init__(self, path: str, conf: Dict[str, str], **kwargs):
        super().__init__(path, conf, type=InferType.CLASSIFICATION, add_post_restore=False, load_strict=False, **kwargs)

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

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        return d

    def is_valid(self) -> bool:
        return False
