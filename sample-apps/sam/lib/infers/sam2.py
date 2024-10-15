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
from typing import Any, Dict, Tuple, Union

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType

logger = logging.getLogger(__name__)


class Sam2(InferTask):
    def __init__(self):
        super().__init__(InferType.ANNOTATION, None, 3, "SAM V2", None)

        # Download PreTrained Model
        # https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description
        # self.PRE_TRAINED_PATH = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        # url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
        # download_file(url, self.path[0])

    def is_valid(self) -> bool:
        return True

    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        return "", {}
