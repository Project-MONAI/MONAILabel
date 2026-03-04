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
        label_tag = request.get("label_tag")
        labels = request.get("labels")
        images = datastore.get_unlabeled_images(label_tag, labels)
        if not len(images):
            return None

        strategy = request["strategy"]
        images_info = []
        for image in images:
            images_info.append(datastore.get_image_info(image).get("strategy", {}).get(strategy, {}))

        current_ts = int(time.time())
        # Clamp to zero so future/corrupt timestamps don't produce negative weights,
        # which would cause random.choices to raise a ValueError.
        weights = [max(0, current_ts - info.get("ts", 0)) for info in images_info]

        if sum(weights) == 0:
            # All images were seen at the current second (or have corrupt timestamps);
            # fall back to a uniform random pick.
            selected_idx = random.randrange(len(images))
        else:
            selected_idx = random.choices(range(len(images)), weights=weights, k=1)[0]
        image = images[selected_idx]
        selected_weight = weights[selected_idx]

        logger.info(f"Random: Selected Image: {image}; Weight: {selected_weight}")

        # If the datastore contains 4d images send the multichannel flag to ensure images are loaded as sequences
        if datastore.get_is_multichannel():
            return {"id": image, "weight": selected_weight, "multichannel": True}

        # If the datastore is multi_file, each sample has a directory with multiple images
        if datastore.get_is_multi_file():
            return {
                "id": image,
                "weight": selected_weight,
                "multi_file": True,
            }  # this will send the directory and we will walk it later on

        return {"id": image, "weight": selected_weight}
