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
import random

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

logger = logging.getLogger(__name__)


class TTAStrategy(Strategy):
    """
    Test Time Augmentation (TTA) as active learning strategy
    """

    def __init__(self):
        super().__init__("Get First Sample Based on TTA score")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        tta_scores = {image: datastore.get_image_info(image).get("vvc_tta", 0) for image in images}

        # PICK RANDOM IF THERE IS NOT VVC_TTA SCORES!!
        if sum(tta_scores.values()) == 0:
            image = random.choice(images)
            logger.info(f"Random: Selected Image: {image}")
        else:
            _, image = max(zip(tta_scores.values(), tta_scores.keys()))
            logger.info(f"TTA: Selected Image: {image}")
        return image
