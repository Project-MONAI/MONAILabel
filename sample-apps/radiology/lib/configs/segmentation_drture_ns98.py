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

        # REMEMBER: DO NOT USE largest connected components transform for inference!
        self.labels = {
            "Left-Cerebral-White-Matter": 1,
            "Left-Lateral-Ventricle": 2,
            "Left-Inf-Lat-Vent": 3,
            "Left-Cerebellum-White-Matter": 4,
            "Left-Cerebellum-Cortex": 5,
            "Left-Thalamus": 6,
            "Left-Caudate": 7,
            "Left-Putamen": 8,
            "Left-Pallidum": 9,
            "3rd-Ventricle": 10,
            "4th-Ventricle": 11,
            "Brain-Stem": 12,
            "Left-Hippocampus": 13,
            "Left-Amygdala": 14,
            "CSF": 15,
            "Left-Accumbens-area": 16,
            "Left-VentralDC": 17,
            "Right-Cerebral-White-Matter": 18,
            "Right-Lateral-Ventricle": 19,
            "Right-Inf-Lat-Vent": 20,
            "Right-Cerebellum-White-Matter": 21,
            "Right-Cerebellum-Cortex": 22,
            "Right-Thalamus": 23,
            "Right-Caudate": 24,
            "Right-Putamen": 25,
            "Right-Pallidum": 26,
            "Right-Hippocampus": 27,
            "Right-Amygdala": 28,
            "Right-Accumbens-area": 29,
            "Right-VentralDC": 30,
            "ctx-lh-bankssts": 31,
            "ctx-lh-caudalanteriorcingulate": 32,
            "ctx-lh-caudalmiddlefrontal": 33,
            "ctx-lh-cuneus": 34,
            "ctx-lh-entorhinal": 35,
            "ctx-lh-fusiform": 36,
            "ctx-lh-inferiorparietal": 37,
            "ctx-lh-inferiortemporal": 38,
            "ctx-lh-isthmuscingulate": 39,
            "ctx-lh-lateraloccipital": 40,
            "ctx-lh-lateralorbitofrontal": 41,
            "ctx-lh-lingual": 42,
            "ctx-lh-medialorbitofrontal": 43,
            "ctx-lh-middletemporal": 44,
            "ctx-lh-parahippocampal": 45,
            "ctx-lh-paracentral": 46,
            "ctx-lh-parsopercularis": 47,
            "ctx-lh-parsorbitalis": 48,
            "ctx-lh-parstriangularis": 49,
            "ctx-lh-pericalcarine": 50,
            "ctx-lh-postcentral": 51,
            "ctx-lh-posteriorcingulate": 52,
            "ctx-lh-precentral": 53,
            "ctx-lh-precuneus": 54,
            "ctx-lh-rostralanteriorcingulate": 55,
            "ctx-lh-rostralmiddlefrontal": 56,
            "ctx-lh-superiorfrontal": 57,
            "ctx-lh-superiorparietal": 58,
            "ctx-lh-superiortemporal": 59,
            "ctx-lh-supramarginal": 60,
            "ctx-lh-frontalpole": 61,
            "ctx-lh-temporalpole": 62,
            "ctx-lh-transversetemporal": 63,
            "ctx-lh-insula": 64,
            "ctx-rh-bankssts": 65,
            "ctx-rh-caudalanteriorcingulate": 66,
            "ctx-rh-caudalmiddlefrontal": 67,
            "ctx-rh-cuneus": 68,
            "ctx-rh-entorhinal": 69,
            "ctx-rh-fusiform": 70,
            "ctx-rh-inferiorparietal": 71,
            "ctx-rh-inferiortemporal": 72,
            "ctx-rh-isthmuscingulate": 73,
            "ctx-rh-lateraloccipital": 74,
            "ctx-rh-lateralorbitofrontal": 75,
            "ctx-rh-lingual": 76,
            "ctx-rh-medialorbitofrontal": 77,
            "ctx-rh-middletemporal": 78,
            "ctx-rh-parahippocampal": 79,
            "ctx-rh-paracentral": 80,
            "ctx-rh-parsopercularis": 81,
            "ctx-rh-parsorbitalis": 82,
            "ctx-rh-parstriangularis": 83,
            "ctx-rh-pericalcarine": 84,
            "ctx-rh-postcentral": 85,
            "ctx-rh-posteriorcingulate": 86,
            "ctx-rh-precentral": 87,
            "ctx-rh-precuneus": 88,
            "ctx-rh-rostralanteriorcingulate": 89,
            "ctx-rh-rostralmiddlefrontal": 90,
            "ctx-rh-superiorfrontal": 91,
            "ctx-rh-superiorparietal": 92,
            "ctx-rh-superiortemporal": 93,
            "ctx-rh-supramarginal": 94,
            "ctx-rh-frontalpole": 95,
            "ctx-rh-temporalpole": 96,
            "ctx-rh-transversetemporal": 97,
            "ctx-rh-insula": 98,
            "Brain tumor + necrotic": 99,
            "Edema": 100,
            "unknown label": 101,  # Check which number should we associate to this.
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
            url = f"{self.PRE_TRAINED_PATH}/segmentation_unet_drture_ns98.pt"
            download_file(url, self.path[0])

        # Network
        self.spatial_size = json.loads(self.conf.get("spatial_size", "[64, 64, 64]"))
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
            channels=(16, 32, 64),
            strides=(2, 2),
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
