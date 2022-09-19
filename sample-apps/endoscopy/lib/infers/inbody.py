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
import torch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import AsChannelFirstd, AsDiscreted, CastToTyped, NormalizeIntensityd, Resized

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.pre import LoadImageExd
from monailabel.transform.writer import ClassificationWriter

logger = logging.getLogger(__name__)


class InBody(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation model for Tool Tracking.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.CLASSIFICATION,
        labels=None,
        dimension=2,
        description="A pre-trained semantic classification model for InBody/OutBody",
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
            Resized(keys="image", spatial_size=(256, 256), mode="bilinear"),
            CastToTyped(keys="image", dtype=torch.float32),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = ClassificationWriter(label=self.output_label_key, label_names={v: k for k, v in self.labels.items()})
        return writer(data)
