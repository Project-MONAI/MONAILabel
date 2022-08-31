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
from distutils.util import strtobool
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from lib.scoring.cvat import CVATRandomScoring
from monai.networks.nets import SEResNet50

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)


class DeID(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {"Tool": 0}

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/endoscopy_deid.pt"
            download_file(url, self.path[0])

        # Network
        self.network = SEResNet50(spatial_dims=2, in_channels=3, num_classes=2)
        self.network_with_dropout = SEResNet50(spatial_dims=2, in_channels=3, num_classes=2, dropout_prob=0.2)

        # Others
        self.epistemic_enabled = bool(strtobool(conf.get("epistemic_enabled", "false")))
        self.epistemic_enabled = self.epistemic_enabled if self.conf.get("models") == "deid" else False
        self.epistemic_max_samples = int(conf.get("epistemic_max_samples", "0"))
        self.epistemic_simulation_size = int(conf.get("epistemic_simulation_size", "5"))

        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_max_samples}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.DeID(
            path=self.path,
            network=self.network,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        task: TrainTask = lib.trainers.DeID(
            model_dir=output_dir,
            network=self.network,
            load_path=self.path[0],
            publish_path=self.path[1],
            labels=self.labels,
            description="Train DeID Model",
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
            methods[f"{self.name}_epistemic"] = CVATRandomScoring(
                top_k=int(self.conf.get("epistemic_top_k", "10")), function="monailabel.endoscopy.deid"
            )
        return methods
