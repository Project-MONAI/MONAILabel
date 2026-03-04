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
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import SegResNet
from monai.utils import optional_import

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

_, has_cp = optional_import("cupy")
_, has_cucim = optional_import("cucim")

logger = logging.getLogger(__name__)


class Segmentation(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        """Initializes the SegmentationBrats task."""
        super().init(name, model_dir, conf, planner, **kwargs)

        # BraTS labels: 3 multi-label channels produced by ConvertToMultiChannelBasedOnBratsClassesd
        #   Channel 0: TC  - Tumor Core        (label 2 OR label 3)
        #   Channel 1: WT  - Whole Tumor       (label 1 OR label 2 OR label 3)
        #   Channel 2: ET  - Enhancing Tumor   (label 2)
        self.labels = {
            "tumor core": 1,  # Tumor Core
            "whole tumor": 2,  # Whole Tumor
            "enhancing tumor": 3,  # Enhancing Tumor
        }

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model (optional)
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_segmentation_segresnet_brats.pt"
            download_file(url, self.path[0])

        # Spacing and ROI for BraTS (isotropic 1mm, large crop matching tutorial)
        self.target_spacing = (1.0, 1.0, 1.0)
        self.roi_size = (224, 224, 144)

        # Number of input channels: 4 MRI modalities (FLAIR, T1, T1Gd, T2)
        # when multi_file=True the LoadDirectoryImagesd loader stacks them;
        # when multi_file=False the image file must already be a 4-channel volume.
        try:
            input_channels = int(self.conf.get("input_channels", 4))
        except (ValueError, TypeError):
            logger.warning("Could not parse input_channels, defaulting to 4")
            input_channels = 4

        # Network
        self.network = SegResNet(
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            init_filters=16,
            in_channels=input_channels,
            out_channels=len(self.labels),  # TC, WT, ET — sigmoid multilabel, no background channel
            dropout_prob=0.2,
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        """Creates the SegmentationBrats InferTask task."""
        task: InferTask = lib.infers.SegmentationBrats(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
            config={"largest_cc": True if has_cp and has_cucim else False},
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        """Creates the SegmentationBrats Trainer task."""
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.SegmentationBrats(
            model_dir=output_dir,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            load_path=load_path,
            publish_path=self.path[1],
            description="Train BraTS Segmentation Model (TC/WT/ET multilabel)",
            labels=self.labels,
        )
        return task
