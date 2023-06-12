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
# TODO: Add SAM to monai networks
from monai.networks.nets import SAM

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class SegmentationSamobject(TaskConfig):
    def __init__(self):
        super().__init__()

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "object": 1,
        }

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            # TODO add SAM pretrained to the path https://github.com/Project-MONAI/MONAILabel/releases/download/pretrained
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            # TODO SAM has some different pretrained models
            url = f"{url}/zeroshot2d_segmentation_sam.pt"
            download_file(url, self.path[0])

        # Network
        self.network = SAM(
            device="cuda",
            model_type="vit_b", # TODO: give options
            checkpoint=self.path[0] # pretrained
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationSamobject(
            path=self.path,
            network=self.network,
            target_spacing=self.target_spacing,
            labels=self.labels,
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.SegmentationSamobject(
            model_dir=output_dir,
            network=self.network,
            description="Train SAM (single object) Segmentation Model",
            load_path=load_path,
            publish_path=self.path[1],
            labels=self.labels,
        )
        return task

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=SAM(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=2,
                    channels=[16, 32, 64, 128, 256],
                    strides=[2, 2, 2, 2],
                    num_res_units=2,
                    norm="batch",
                    dropout=0.2,
                ),
                transforms=lib.infers.SegmentationSpleen(None).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        return methods
