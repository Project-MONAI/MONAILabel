# Copyright 2020 - 2021 MONAI Consortium
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

import numpy as np
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType

logger = logging.getLogger(__name__)


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        patch_size=(256, 256),
        type=InferType.SEGMENTATION,
        labels=(
            "tumor",
            "stroma",
            "lymphocytic_infiltrate",
            "necrosis_or_debris",
            "glandular_secretions",
            "blood",
            "exclude",
            "metaplasia_NOS",
            "fat",
            "plasma_cells",
            "other_immune_infiltrate",
            "mucoid_material",
            "normal_acinus_or_duct",
            "lymphatics",
            "undetermined",
            "nerve",
            "skin_adnexa",
            "blood_vessel",
            "angioinvasion",
            "dcis",
            "other",
        ),
        dimension=2,
        description="A pre-trained model Pathology",
    ):
        self.patch_size = patch_size
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys="image", dtype=np.uint8),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys="image"),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=self.patch_size)

    def post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            ToNumpyd(keys="pred"),
        ]
