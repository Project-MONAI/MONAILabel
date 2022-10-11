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
from abc import ABCMeta

from monai.utils import deprecated

from monailabel.interfaces.tasks.infer_v2 import InferType as InferTypeV2
from monailabel.tasks.infer.basic_infer import BasicInferTask

logger = logging.getLogger(__name__)

# Alias
InferType = InferTypeV2


@deprecated(since="0.5.2", msg_suffix="please use monailabel.tasks.infer.basic_infer.BasicInferTask instead")
class InferTask(BasicInferTask, metaclass=ABCMeta):
    pass
