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
from lib.transforms import LoadImagePatchd
from monai.apps.nuclick.transforms import AddClickSignalsd, PostFilterLabeld
from monai.transforms import Activationsd, AsChannelFirstd, AsDiscreted, EnsureTyped, SqueezeDimd, ToNumpyd

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class NuClick(InferTask):
    """
    This provides Inference Engine for pre-trained NuClick segmentation (UNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(128, 128),
        type=InferType.OTHERS,
        labels=None,
        dimension=2,
        description="A pre-trained NuClick model for interactive cell segmentation for Pathology",
        **kwargs,
    ):
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

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        d["nuclick"] = True
        return d

    def pre_transforms(self, data=None):
        return [
            LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False),
            AsChannelFirstd(keys="image"),
            AddClickSignalsd(image="image"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        return super().run_inferer(data, False, device)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            SqueezeDimd(keys="pred", dim=1),
            ToNumpyd(keys=("image", "pred")),
            PostFilterLabeld(keys="pred"),
            FindContoursd(keys="pred", labels=self.labels),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
