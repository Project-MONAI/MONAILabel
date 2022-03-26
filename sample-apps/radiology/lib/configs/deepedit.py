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

import json
import logging
import os
from distutils.util import strtobool
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import UNETR, DynUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)


class DeepEdit(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Multilabel
        # self.labels = {
        #     "spleen": 1,
        #     "right kidney": 2,
        #     "left kidney": 3,
        #     "liver": 6,
        #     "stomach": 7,
        #     "aorta": 8,
        #     "inferior vena cava": 9,
        #     "background": 0,
        # }

        # Single label
        self.labels = {
            "spleen": 1,
            "background": 0,
        }

        network = self.conf.get("network", "dynunet")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{self.name}_{network}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.PRE_TRAINED_PATH}/deepedit_{network}_singlelabel.pt"
            download_file(url, self.path[0])

        # Network
        self.spatial_size = json.loads(self.conf.get("spatial_size", "[128, 128, 128]"))
        if network == "unetr":
            self.network = UNETR(
                spatial_dims=3,
                in_channels=len(self.labels) + 1,
                out_channels=len(self.labels),
                img_size=self.spatial_size,
                feature_size=64,
                hidden_size=1536,
                mlp_dim=3072,
                num_heads=48,
                pos_embed="conv",
                norm_name="instance",
                res_block=True,
            )
        else:
            self.network = DynUNet(
                spatial_dims=3,
                in_channels=len(self.labels) + 1,
                out_channels=len(self.labels),
                kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
                upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
            )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: lib.infers.DeepEdit(
                path=self.path, network=self.network, labels=self.labels, spatial_size=self.spatial_size
            ),
            f"{self.name}_seg": lib.infers.DeepEdit(
                path=self.path,
                network=self.network,
                labels=self.labels,
                spatial_size=self.spatial_size,
                type=InferType.SEGMENTATION,
            ),
        }

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, f"{self.name}_" + self.conf.get("network", "dynunet"))
        task: TrainTask = lib.trainers.DeepEdit(
            model_dir=output_dir,
            network=self.network,
            load_path=self.path[0],
            publish_path=self.path[1],
            spatial_size=self.planner.spatial_size if self.planner else (128, 128, 128),
            target_spacing=self.planner.target_spacing if self.planner else (1.0, 1.0, 1.0),
            config={"pretrained": strtobool(self.conf.get("use_pretrained_model", "true"))},
            labels=self.labels,
            debug_mode=False,
            find_unused_parameters=True,
        )
        return task
