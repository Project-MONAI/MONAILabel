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
from monai.networks.nets import UNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)


class SegmentationBrats(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "Cerebral white matter": 1,
            "Cerebral cortex": 2,
            "Ventricles": 3,
            "Cerebellum white matter": 4,
            "Cerebellum cortex": 5,
            "Thalamus": 6,
            "Caudate": 7,
            "Putamen": 8,
            "Pallidum": 9,
            "Brain-stem": 10,
            "Hippocampus": 11,
            "Amygdala": 12,
            "Accumbens area": 13,
            "Ventral diencephalon": 14,
            "Brain tumor + necrotic": 15,
            "Edema": 16,
        }

        # Number of input channels - 4 for BRATS and 1 for spleen
        self.number_intensity_ch = 4

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            url = f"{self.PRE_TRAINED_PATH}/segmentation_unet_brats_ns16.pt"
            download_file(url, self.path[0])

        # Network
        self.spatial_size = json.loads(self.conf.get("spatial_size", "[96, 96, 96]"))
        # self.network = UNETR(
        #     spatial_dims=3,
        #     in_channels=self.number_intensity_ch,
        #     out_channels=len(self.labels) + 1,  # labels plus background
        #     img_size=self.spatial_size,
        #     feature_size=64,
        #     hidden_size=1536,
        #     mlp_dim=3072,
        #     num_heads=48,
        #     pos_embed="conv",
        #     norm_name="instance",
        #     res_block=True,
        # )

        # self.network = SwinUNETR(
        #     img_size=self.spatial_size,
        #     in_channels=self.number_intensity_ch,
        #     out_channels=len(self.labels) + 1,  # labels plus background
        #     feature_size=48,
        #     drop_rate=0.0,
        #     attn_drop_rate=0.0,
        #     dropout_path_rate=0.0,
        #     use_checkpoint=True,
        # )

        self.network = UNet(
            spatial_dims=3,
            in_channels=self.number_intensity_ch,
            out_channels=len(self.labels) + 1,  # labels plus background,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.2,
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationBrats(
            path=self.path,
            network=self.network,
            spatial_size=self.spatial_size,
            labels=self.labels,
            config={"largest_cc": True},
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        task: TrainTask = lib.trainers.SegmentationBrats(
            model_dir=output_dir,
            network=self.network,
            spatial_size=self.spatial_size,
            load_path=self.path[0],
            publish_path=self.path[1],
            description="Train Multilabel Segmentation Model",
            dimension=3,
            labels=self.labels,
            find_unused_parameters=True,
        )
        return task
