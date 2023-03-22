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
from typing import Dict, Optional

from monailabel.tasks.train.bundle import BundleConstants, BundleTrainTask

logger = logging.getLogger(__name__)


class ToolTracking(BundleTrainTask):
    def __init__(self, path: str, conf: Dict[str, str], const: Optional[BundleConstants] = None):
        super().__init__(path, conf, const, enable_tracking=True)