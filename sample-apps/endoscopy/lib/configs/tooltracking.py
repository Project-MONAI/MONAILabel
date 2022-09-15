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
from lib.scoring.cvat import CVATEpistemicScoring
from monai.networks.nets import FlexibleUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class ToolTracking(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {"Tool": 1}
        self.label_colors = {"Tool": (255, 0, 0)}

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/endoscopy_tooltracking.pt"
            try:
                download_file(url, self.path[0])
            except:
                logger.warning(f"Failed to download pre-trained model from {url}; Ignoring the same...")

        # Network
        self.network = FlexibleUNet(
            in_channels=3,
            out_channels=2,
            backbone="efficientnet-b0",
            spatial_dims=2,
            is_pad=True,
            pretrained=True,
        )
        self.network_with_dropout = FlexibleUNet(
            in_channels=3,
            out_channels=2,
            backbone="efficientnet-b0",
            spatial_dims=2,
            is_pad=True,
            pretrained=True,
            dropout=0.2,
        )

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_max_samples = int(conf.get("epistemic_max_samples", "0"))
        self.epistemic_simulation_size = int(conf.get("epistemic_simulation_size", "5"))

        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_max_samples}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.ToolTracking(
            path=self.path,
            network=self.network,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
            config={
                "label_colors": self.label_colors,
            },
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        task: TrainTask = lib.trainers.ToolTracking(
            model_dir=output_dir,
            network=self.network,
            load_path=self.path[0],
            publish_path=self.path[1],
            labels=self.labels,
            description="Train Tool Tracking Model",
            config={
                "max_epochs": 10,
                "train_batch_size": 1,
                "val_split": 0,
            },
            find_unused_parameters=True,
        )
        return task

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {}

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = CVATEpistemicScoring(
                top_k=int(self.conf.get("epistemic_top_k", "10")),
                infer_task=lib.infers.ToolTracking(
                    path=self.path,
                    network=self.network_with_dropout,
                    labels=self.labels,
                    train_mode=True,
                    skip_writer=True,
                    find_contours=False,
                ),
                function="monailabel.endoscopy.tooltracking",
                max_samples=self.epistemic_max_samples,
                simulation_size=self.epistemic_simulation_size,
                use_variance=True,
            )
        return methods
