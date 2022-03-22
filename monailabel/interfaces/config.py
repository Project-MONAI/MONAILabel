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
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class TaskConfig(metaclass=ABCMeta):
    PRE_TRAINED_PATH = "https://github.com/Project-MONAI/MONAILabel/releases/download/data"
    NGC_PATH = "https://api.ngc.nvidia.com/v2/models/nvidia/med"

    def __init__(self):
        self.name = None
        self.model_dir = None
        self.conf = None
        self.planner = None
        self.kwargs = None

        self.network = None
        self.path = None
        self.labels = None
        self.label_colors = None

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        self.name = name
        self.model_dir = model_dir
        self.conf = conf
        self.planner = planner
        self.kwargs = kwargs

    @abstractmethod
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        pass

    @abstractmethod
    def trainer(self) -> Optional[TrainTask]:
        pass

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        return None

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        return None
