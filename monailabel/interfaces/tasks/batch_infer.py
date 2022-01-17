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

import copy
import logging
from enum import Enum
from typing import Callable

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.utils.others.generic import remove_file

logger = logging.getLogger(__name__)


class BatchInferImageType(str, Enum):
    IMAGES_ALL = "all"
    IMAGES_LABELED = "labeled"
    IMAGES_UNLABELED = "unlabeled"


class BatchInferTask:
    """
    Basic Batch Infer Task
    """

    def get_images(self, request, datastore: Datastore):
        """
        Override this method to get all eligible images for your task to run batch infer
        """
        images = request.get("images", BatchInferImageType.IMAGES_ALL)
        if isinstance(images, str):
            if images == BatchInferImageType.IMAGES_LABELED:
                return datastore.get_labeled_images()
            if images == BatchInferImageType.IMAGES_UNLABELED:
                return datastore.get_unlabeled_images()
            return datastore.list_images()
        return images

    def __call__(self, request, datastore: Datastore, infer: Callable):
        image_ids = self.get_images(request, datastore)
        logger.info(f"Total number of images for batch inference: {len(image_ids)}")

        result = {}
        for image_id in image_ids:
            req = copy.deepcopy(request)
            req["image"] = image_id
            req["save_label"] = True
            req["label_tag"] = DefaultLabelTag.ORIGINAL

            logger.info(f"Running inference for image id {image_id}")
            r = infer(req, datastore)
            if r.get("file"):
                remove_file(r.get("file"))
            result["image_id"] = r
        return result
