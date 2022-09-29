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

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.activelearning.random import Random

logger = logging.getLogger(__name__)


class WSIRandom(Random):
    """
    Consider implementing a random strategy for active learning for WSI Images
    """

    def __init__(self):
        super().__init__()
        self.description = "Random Strategy for WSI Images/Patches"

    def __call__(self, request, datastore: Datastore):
        image = request.get("image")

        # Fetch Random Image
        if not image:
            return super().__call__(request, datastore)

        # Fetch Random Patch
        image_size = request.get("image_size", [0, 0])
        patch_size = request.get("patch_size", [1024, 1024])

        image_size[0] = patch_size[0] if image_size[0] <= 0 else image_size[0]
        image_size[1] = patch_size[1] if image_size[1] <= 0 else image_size[1]

        max_x = max(0, image_size[0] - patch_size[0])
        max_y = max(0, image_size[1] - patch_size[1])

        x = random.randint(0, max_x - 1)
        y = random.randint(0, max_y - 1)
        w = min(image_size[0], patch_size[0])
        h = min(image_size[1], patch_size[1])

        if x + w > image_size[0]:
            x = x - (w - image_size[0])
        if y + h > image_size[1]:
            y = y - (h - image_size[1])

        bbox = [x, y, w, h]
        logger.info(f"Using BBOX: {bbox}")
        return {"id": image, "bbox": bbox}
