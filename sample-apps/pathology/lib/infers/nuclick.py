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
from typing import Any, Callable, Dict, Sequence

import numpy as np
import torch
from lib.transforms import FixNuclickClassd, LoadImagePatchd, NuClickPostFilterLabelExd
from monai.apps.nuclick.transforms import AddClickSignalsd, NuclickKeys
from monai.config import KeysCollection
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    EnsureTyped,
    MapTransform,
    SqueezeDimd,
    ToNumpyd,
)
from monai.utils import ensure_tuple

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class ConvertInteractiveClickSignals(MapTransform):
    """
    ConvertInteractiveClickSignals converts interactive annotation information (e.g. from DSA) into a format expected
    by NuClick. Typically, it will take point annotations from data["annotations"][<source_annotation_key>], convert
    it to 2d points, and place it in data[<target_data_key>].
    """

    def __init__(
        self, source_annotation_keys: KeysCollection, target_data_keys: KeysCollection, allow_missing_keys: bool = False
    ):
        super().__init__(target_data_keys, allow_missing_keys)
        self.source_annotation_keys = ensure_tuple(source_annotation_keys)
        self.target_data_keys = ensure_tuple(target_data_keys)

    def __call__(self, data):
        data = dict(data)
        annotations = data.get("annotations", {})
        annotations = {} if annotations is None else annotations
        for source_annotation_key, target_data_key in zip(self.source_annotation_keys, self.target_data_keys):
            if source_annotation_key in annotations:
                points = annotations.get(source_annotation_key)["points"]
                print(f"points={points}")
                points = [coords[0:2] for coords in points]
                data[target_data_key] = points
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"source_annotation_key={source_annotation_key} not found in annotations.keys()={annotations.keys()}"
                )
        return data


class NuClick(BasicInferTask):
    """
    This provides Inference Engine for pre-trained NuClick segmentation (UNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(128, 128),
        type=InferType.ANNOTATION,
        labels=None,
        dimension=2,
        description="A pre-trained NuClick model for interactive cell segmentation for Pathology",
        **kwargs,
    ):
        self.task_classification = None
        super().__init__(
            path=path,
            network=network,
            roi_size=roi_size,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

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

    def pre_transforms(self, data=None):
        return [
            LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False),
            AsChannelFirstd(keys="image"),
            ConvertInteractiveClickSignals(
                source_annotation_keys="nuclick points",
                target_data_keys=NuclickKeys.FOREGROUND,
                allow_missing_keys=True,
            ),
            AddClickSignalsd(image="image", foreground=NuclickKeys.FOREGROUND),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        output = super().run_inferer(data, False, device)
        if self.task_classification:
            data2 = copy.deepcopy(self.task_classification.config())
            pred1 = output["pred"]
            pred1 = torch.sigmoid(pred1)
            pred1 = pred1 >= 0.5

            data2.update({"image": output["image"][:, :3], "label": pred1, "device": device})

            data2 = self.task_classification.run_pre_transforms(data2, [FixNuclickClassd(image="image", label="label")])

            output2 = self.task_classification.run_inferer(data2, False, device)
            pred2 = output2["pred"]
            pred2 = torch.softmax(pred2, dim=1)
            pred2 = torch.argmax(pred2, dim=1)
            pred2 = [int(p) for p in pred2]

            output["pred_classes"] = [v + 1 for v in pred2]
            logger.info(f"Predicted Classes: {output['pred_classes']}")
        return output

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            SqueezeDimd(keys="pred", dim=1),
            ToNumpyd(keys=("image", "pred")),
            NuClickPostFilterLabelExd(keys="pred"),
            FindContoursd(keys="pred", labels=self.labels),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
