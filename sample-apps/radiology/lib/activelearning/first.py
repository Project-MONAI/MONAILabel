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

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.tasks.strategy import Strategy, DefaultAnnotationMode

logger = logging.getLogger(__name__)


class First(Strategy):
    """
    Consider implementing a first strategy for active learning
    """

    def __init__(self, annotation_mode: str = DefaultAnnotationMode.COLLABORATIVE):
        super().__init__("Get First Sample", annotation_mode)

    def __call__(self, request, datastore: Datastore):
        if self.annotation_mode == DefaultAnnotationMode.COMPETETIVE:
            tag = request.get('client_id', DefaultLabelTag.FINAL)
        else:
             tag = DefaultLabelTag.FINAL
        images = datastore.get_unlabeled_images(tag)
        if not len(images):
            return None

        images.sort()
        image = images[0]

        logger.info(f"First: Selected Image: {image}")
        return image
