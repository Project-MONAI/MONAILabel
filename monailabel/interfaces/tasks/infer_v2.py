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
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, Dict, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class InferType(str, Enum):
    """
    Type of Inference Model

    Attributes:
        SEGMENTATION -            Segmentation Model
        ANNOTATION -              Annotation Model
        CLASSIFICATION -          Classification Model
        DEEPGROW -                Deepgrow Interactive Model
        DEEPEDIT -                DeepEdit Interactive Model
        SCRIBBLES -               Scribbles Model
        DETECTION -               Detection Model
        OTHERS -                  Other Model Type
    """

    SEGMENTATION: str = "segmentation"
    ANNOTATION: str = "annotation"
    CLASSIFICATION: str = "classification"
    DEEPGROW: str = "deepgrow"
    DEEPEDIT: str = "deepedit"
    SCRIBBLES: str = "scribbles"
    DETECTION: str = "detection"
    OTHERS: str = "others"


class InferTask(metaclass=ABCMeta):
    """
    Inference Task
    """

    def __init__(
        self,
        type: Union[str, InferType],
        labels: Union[str, None, Sequence[str], Dict[Any, Any]],
        dimension: int,
        description: str,
        config: Union[None, Dict[str, Any]] = None,
    ):
        """
        :param type: Type of Infer (segmentation, deepgrow etc..)
        :param labels: Labels associated to this Infer
        :param dimension: Input dimension
        :param description: Description
        :param config: K,V pairs to be part of user config
        """

        self.type = type
        self.labels = [] if labels is None else [labels] if isinstance(labels, str) else labels
        self.dimension = dimension
        self.description = description

        self._config: Dict[str, Any] = {}
        if config:
            self._config.update(config)

    def info(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "labels": self.labels,
            "dimension": self.dimension,
            "description": self.description,
            "config": self.config(),
        }

    def config(self) -> Dict[str, Any]:
        return self._config

    def get_path(self, validate=True):
        return None

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        pass
