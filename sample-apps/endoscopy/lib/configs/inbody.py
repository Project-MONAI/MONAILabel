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
from lib.scoring.cvat import CVATRandomScoring
from monai.bundle import download

from monailabel.config import settings
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.utils.others.generic import strtobool

logger = logging.getLogger(__name__)


class InBody(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        bundle_name = "endoscopic_inbody_classification"
        version = conf.get("inbody", "0.4.4")
        zoo_source = conf.get("zoo_source", settings.MONAI_ZOO_SOURCE)

        self.bundle_path = os.path.join(self.model_dir, bundle_name)
        if not os.path.exists(self.bundle_path):
            download(name=bundle_name, version=version, bundle_dir=self.model_dir, source=zoo_source)

        # Others
        self.epistemic_enabled = bool(strtobool(conf.get("epistemic_enabled", "false")))
        self.epistemic_enabled = self.epistemic_enabled if self.conf.get("models") == name else False
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.InBody(self.bundle_path, self.conf)
        return task

    def trainer(self) -> Optional[TrainTask]:
        task: TrainTask = lib.trainers.InBody(self.bundle_path, self.conf)
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
                top_k=int(self.conf.get("epistemic_top_k", "10")), function="monailabel.endoscopy.inbody"
            )
        return methods
