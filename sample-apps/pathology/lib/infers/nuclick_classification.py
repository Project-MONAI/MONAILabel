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
import copy
import logging

from lib.infers import NuClick
from lib.transforms import FixNuclickClassd
from monailabel.tasks.infer.basic_infer import BasicInferTask

logger = logging.getLogger(__name__)


class NuClickClassification(NuClick):
    def __init__(
            self,
            **kwargs,
    ):
        self.task_classification = None
        super().__init__(**kwargs)

    def init_classification(self, task_classification: BasicInferTask):
        self.task_classification = task_classification
        self.labels = task_classification.labels
        self.description = "Combines Nuclick and Classification"

    def is_valid(self) -> bool:
        return super().is_valid() and (not self.task_classification or self.task_classification.is_valid())

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        output = super().run_inferer(data, False, device)
        if self.task_classification:
            data2 = copy.deepcopy(self.task_classification.config())
            data2.update({"image": output["image"][:, :3], "label": output["pred"], "device": device})

            data2 = self.task_classification.run_pre_transforms(data2, [FixNuclickClassd(image="image", label="label")])

            output2 = self.task_classification.run_inferer(data2, False, device)
            output2 = self.task_classification.run_post_transforms(
                output2, self.task_classification.post_transforms(output2)
            )
            logger.info(output2["pred"])
            output["classification"] = output2
        return output
