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
from lib.transforms import LoadImageExd
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    AsChannelFirstd,
    AsDiscreted,
    DivisiblePadd,
    EnsureTyped,
    Resized,
    ScaleIntensityd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import FindContoursd, Restored
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class ToolTracking(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation model for Tool Tracking.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained semantic segmentation model for Tool Tracking",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["endoscopy"] = True
        return d

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImageExd(keys="image", dtype=np.uint8),
            AsChannelFirstd(keys="image"),
            Resized(keys="image", spatial_size=(736, 480)),
            DivisiblePadd(keys="image", k=32),
            ScaleIntensityd(keys="image"),
            ToTensord(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred", dtype=np.uint8),
            Restored(keys="pred", ref_image="image"),
            SqueezeDimd(keys="pred", dim=0),
            FindContoursd(keys="pred", labels=self.labels),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
