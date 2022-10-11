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
from monai.networks.nets import BasicUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class Deepgrow3D(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = [
            "spleen",
            "right kidney",
            "left kidney",
            "liver",
            "stomach",
            "aorta",
            "inferior vena cava",
        ]

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/deepgrow_3d_bunet.pt"
            download_file(url, self.path[0])

        # Network
        self.network = BasicUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            features=(32, 64, 128, 256, 512, 32),
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.Deepgrow(
            path=self.path,
            network=self.network,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
            dimension=3,
            model_size=(128, 192, 192),
            config={"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.Deepgrow(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            description="Train 3D Deepgrow model",
            dimension=3,
            labels=self.labels,
            roi_size=(128, 192, 192),
            model_size=(128, 192, 192),
            max_train_interactions=15,
            max_val_interactions=10,
            val_interval=5,
        )
        return task
