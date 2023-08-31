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
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from lib.transforms import ConvertInteractiveClickSignals, LoadImagePatchd
from monai.apps.nuclick.transforms import AddLabelAsGuidanced, NuclickKeys, PostFilterLabeld
from monai.transforms import KeepLargestConnectedComponentd, LoadImaged, SaveImaged, SqueezeDimd

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class NuClick(BundleInferTask):
    """
    This provides Inference Engine for pre-trained NuClick segmentation (UNet) model.
    """

    def __init__(self, path: str, conf: Dict[str, str], **kwargs):
        super().__init__(
            path,
            conf,
            type=InferType.ANNOTATION,
            add_post_restore=False,
            load_strict=False,
            **kwargs,
            pre_filter=[LoadImaged, SqueezeDimd],
            post_filter=[KeepLargestConnectedComponentd, SaveImaged],
        )

        # Override Labels
        self.labels = {"Nuclei": 1}
        self.label_colors = {"Nuclei": (0, 255, 255)}
        self._config["label_colors"] = self.label_colors
        self.task_classification: Optional[BasicInferTask] = None

    def init_classification(self, task_classification: BasicInferTask):
        self.task_classification = task_classification
        self.labels = task_classification.labels
        self._config.update(task_classification._config)
        self.description = (
            "Nuclick with Classification Support using "
            "NuClick (nuclei segmentation) and Segmentation Nuclei (nuclei classification) models"
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        d["nuclick"] = True
        return d

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False),
            ConvertInteractiveClickSignals(
                source_annotation_keys="nuclick points",
                target_data_keys=NuclickKeys.FOREGROUND,
                allow_missing_keys=True,
            ),
        ]
        t.extend([x for x in super().pre_transforms(data)])
        return t

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        output = super().run_inferer(data, False, device)
        if self.task_classification:
            pred1 = output["pred"]
            pred1 = torch.sigmoid(pred1)
            pred1 = pred1 >= 0.5

            data2 = copy.deepcopy(self.task_classification.config())
            data2.update({"image": output["image"][:, :3], "label": pred1, "device": device})
            data2 = self.task_classification.run_pre_transforms(
                data2, [AddLabelAsGuidanced(keys="image", source="label")]
            )

            output2 = self.task_classification.run_inferer(data2, False, device)
            pred2 = output2["pred"]
            pred2 = torch.softmax(pred2, dim=1)
            pred2 = torch.argmax(pred2, dim=1)
            pred2 = [int(p) for p in pred2]

            output[NuclickKeys.PRED_CLASSES] = [v + 1 for v in pred2]
            logger.info(f"Predicted Classes: {output[NuclickKeys.PRED_CLASSES]}")
        return output

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = [x for x in super().post_transforms(data)]
        t.extend(
            [
                SqueezeDimd(keys="pred", dim=1),
                PostFilterLabeld(keys="pred"),
                FindContoursd(keys="pred", labels=self.labels),
            ]
        )
        return t

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
