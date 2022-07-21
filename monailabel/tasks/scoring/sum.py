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

import numpy as np
import torch
from monai.transforms import LoadImage

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.tasks.scoring import ScoringMethod

logger = logging.getLogger(__name__)


class Sum(ScoringMethod):
    """
    Consider implementing simple np sum method of label tags; Also add valid slices that have label mask
    """

    def __init__(self, tags=(DefaultLabelTag.FINAL.value, DefaultLabelTag.ORIGINAL.value)):
        super().__init__("Compute Numpy Sum for Final/Original Labels")
        self.tags = tags

    def __call__(self, request, datastore: Datastore):
        loader = LoadImage(image_only=True)
        result = {}
        for image_id in datastore.list_images():
            for tag in self.tags:
                label_id: str = datastore.get_label_by_image_id(image_id, tag)
                if label_id:
                    label = loader(datastore.get_label_uri(label_id, tag))
                    if isinstance(label, torch.Tensor):
                        label = label.numpy()
                    slices = [sid for sid in range(label.shape[0]) if np.sum(label[sid] > 0)]
                    info = {"sum": int(np.sum(label)), "slices": len(slices)}
                    logger.debug(f"{label_id} => {info}")

                    datastore.update_label_info(label_id, tag, info)
                    result[label_id] = info
        return result
