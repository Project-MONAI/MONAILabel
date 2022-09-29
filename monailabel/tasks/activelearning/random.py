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
import random
import time

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

logger = logging.getLogger(__name__)


class Random(Strategy):
    """
    Consider implementing a random strategy for active learning
    """

    def __init__(self):
        super().__init__("Random Strategy")

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()
        if not len(images):
            return None

        strategy = request["strategy"]
        images_info = []
        for image in images:
            images_info.append(datastore.get_image_info(image).get("strategy", {}).get(strategy, {}))

        current_ts = int(time.time())
        weights = [current_ts - info.get("ts", 0) for info in images_info]

        image = random.choices(images, weights=weights)[0]
        logger.debug(f"Random: Images: {images}; Weight: {weights}")
        logger.info(f"Random: Selected Image: {image}; Weight: {weights[0]}")
        return {"id": image, "weight": weights[0]}
