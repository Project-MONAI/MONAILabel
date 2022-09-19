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

import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import BasicUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class SegmentationNuclei(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "Neoplastic cells": 1,
            "Inflammatory": 2,
            "Connective/Soft tissue cells": 3,
            "Dead Cells": 4,
            "Epithelial": 5,
        }
        self.label_colors = {
            "Neoplastic cells": (255, 0, 0),
            "Inflammatory": (255, 255, 0),
            "Connective/Soft tissue cells": (0, 255, 0),
            "Dead Cells": (0, 0, 0),
            "Epithelial": (0, 0, 255),
        }

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/pathology_segmentation_nuclei.pt"
            download_file(url, self.path[0])

        # Network
        self.network = BasicUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=6,
            features=(32, 64, 128, 256, 512, 32),
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        preload = strtobool(self.conf.get("preload", "false"))
        roi_size = json.loads(self.conf.get("roi_size", "[1024, 1024]"))
        logger.info(f"Using Preload: {preload}; ROI Size: {roi_size}")

        task: InferTask = lib.infers.SegmentationNuclei(
            path=self.path,
            network=self.network,
            labels=self.labels,
            preload=preload,
            roi_size=roi_size,
            config={
                "label_colors": self.label_colors,
                "max_workers": max(1, multiprocessing.cpu_count() // 2),
            },
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.SegmentationNuclei(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            labels=self.labels,
            description="Train Nuclei Segmentation Model",
            config={
                "max_epochs": 10,
                "train_batch_size": 1,
                "dataset_max_region": (10240, 10240),
                "dataset_limit": 0,
                "dataset_randomize": True,
            },
        )
        return task
