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

from typing import Callable, Sequence

import numpy as np
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    ScaleIntensityRangeD,
    SqueezeDimd,
    ToNumpyd,
)

from monailabel.deepedit.transforms import AddClickGuidanced, AddGuidanceSignald, ResizeGuidanced
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import FindContoursd, Restored
from monailabel.transform.pre import LoadImageExd
from monailabel.transform.writer import PolygonWriter


class DeepEdit(InferTask):
    """
    This provides Inference Engine for Deepgrow 2D/3D pre-trained model.
    For More Details, Refer https://github.com/Project-MONAI/tutorials/tree/master/deepgrow/ignite
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        dimension=2,
        description="A pre-trained DeepEdit model based on UNET for Endoscopy",
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

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImageExd(keys="image", dtype=np.uint8),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image", spatial_size=self.roi_size, mode="area"),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            AddClickGuidanced(keys=("foreground", "background"), guidance="guidance"),
            ResizeGuidanced(keys="guidance", ref_image="image"),
            AddGuidanceSignald(keys="image", guidance="guidance"),
        ]

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            Restored(keys="pred", ref_image="image"),
            SqueezeDimd(keys="pred"),
            ToNumpyd(keys="pred", dtype=np.uint8),
            FindContoursd(keys="pred", labels=self.labels, key_foreground_points="foreground"),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
