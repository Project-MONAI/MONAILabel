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
import os
from distutils.util import strtobool
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import UNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)


class SegmentationSpleen(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = ["spleen"]

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.NGC_PATH}/clara_pt_spleen_ct_segmentation/versions/1/files/models/model.pt"
            download_file(url, self.path[0])

        # Network
        self.network = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=[16, 32, 64, 128, 256],
            strides=[2, 2, 2, 2],
            num_res_units=2,
            norm="batch",
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationSpleen(path=self.path, network=self.network, labels=self.labels)
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        task: TrainTask = lib.trainers.SegmentationSpleen(
            model_dir=output_dir,
            network=self.network,
            description="Train Spleen Segmentation Model",
            load_path=self.path[0],
            publish_path=self.path[1],
            labels=self.labels,
        )
        return task
