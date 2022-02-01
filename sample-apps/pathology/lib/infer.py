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

from monai.inferers import SimpleInferer

from monailabel.interfaces.tasks.infer import InferTask, InferType

from .transforms import GridToLabeld, ImageToGridd

logger = logging.getLogger(__name__)


class MyInfer(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        image_size=1024,
        patch_size=64,
        type=InferType.SEGMENTATION,
        labels="tumor",
        dimension=2,
        description="A pre-trained model Pathology",
    ):
        self._image_size = image_size
        self._patch_size = patch_size

        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            ImageToGridd(
                keys="image",
                image_size=self._image_size,
                patch_size=self._patch_size,
                jitter=False,
                flip=False,
                rotate=False,
            ),
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            GridToLabeld(keys="pred", image_size=self._image_size, patch_size=self._patch_size),
        ]

    def run_inferer(self, data, convert_to_batch=True, device="cuda", output_squeezed=False):
        return super().run_inferer(data, convert_to_batch, device, True)
