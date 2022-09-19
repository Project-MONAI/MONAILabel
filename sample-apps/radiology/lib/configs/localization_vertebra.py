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
from monai.networks.nets import UNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class LocalizationVertebra(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "C1": 1,
            "C2": 2,
            "C3": 3,
            "C4": 4,
            "C5": 5,
            "C6": 6,
            "C7": 7,
            "Th1": 8,
            "Th2": 9,
            "Th3": 10,
            "Th4": 11,
            "Th5": 12,
            "Th6": 13,
            "Th7": 14,
            "Th8": 15,
            "Th9": 16,
            "Th10": 17,
            "Th11": 18,
            "Th12": 19,
            "L1": 20,
            "L2": 21,
            "L3": 22,
            "L4": 23,
            "L5": 24,
        }

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/localization_vertebra_unet.pt"
            download_file(url, self.path[0])

        self.target_spacing = (1.3, 1.3, 1.3)  # target space for image
        # Setting ROI size - This is for the image padding
        self.roi_size = (96, 96, 96)

        # Network
        self.network = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(self.labels) + 1,  # labels plus background,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            dropout=0.2,
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.LocalizationVertebra(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.LocalizationVertebra(
            model_dir=output_dir,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            load_path=load_path,
            publish_path=self.path[1],
            description="Train vertebra localization Model",
            dimension=3,
            labels=self.labels,
        )
        return task
