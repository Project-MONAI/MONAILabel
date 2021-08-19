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

import numpy as np

"""
from monai.transforms import LoadImage
"""

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks import ScoringMethod

logger = logging.getLogger(__name__)


class TtaScoring(ScoringMethod):
    """
    First version of test time augmentation active learning
    """

    def __init__(self, tags=("first", "second", "third")):
        super().__init__("Compute initial score based on TTA")
        self.tags = tags

    def __call__(self, request, datastore: Datastore):
        """
        loader = LoadImage(image_only=True)
        avg = []
        """
        result = {}
        for image_id in datastore.list_images():
            """
            for tag in self.tags:
                label_id: str = datastore.get_label_by_image_id(image_id, tag)
                if label_id:
                    aux = np.sum(loader(datastore.get_label_uri(label_id)))
                    avg.append(aux)
            avg = np.array(avg)
            """
            info = {"tta_score": np.random.randint(1, 100)}
            logger.info(f"{image_id} => {info}")
            datastore.update_image_info(image_id, info)
            result[image_id] = info
        return result
