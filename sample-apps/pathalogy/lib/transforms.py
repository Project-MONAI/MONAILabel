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
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform
from PIL import Image
from torchvision.transforms import ColorJitter

logger = logging.getLogger(__name__)


# You can write your transforms here... which can be used in your train/infer tasks
class ImageToGridd(MapTransform, RandomizableTransform):
    def __init__(
        self,
        keys: KeysCollection,
        image_size,
        patch_size,
        brightness=64.0 / 255.0,
        contrast=0.75,
        saturation=0.25,
        hue=0.04,
        label_key="label",
    ):
        super().__init__(keys)

        if image_size % patch_size != 0:
            raise Exception("Image size / patch size != 0 : {} / {}".format(image_size, patch_size))

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_per_side = self.image_size // self.patch_size
        self.grid_size = self.patch_per_side * self.patch_per_side

        self.label_key = label_key
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data):
        d = dict(data)
        flip_right = self.R.uniform()
        num_rotate = self.R.uniform(low=0, high=4)

        for key in self.keys:
            img = Image.open(d[key])

            # jitter (image only)
            if key != self.label_key:
                img = self.color_jitter(img)

            # flip
            if flip_right > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # rotate
            img = img.rotate(90 * num_rotate)

            # to numpy
            img = np.array(img, dtype=np.float32)
            img = img.transpose((2, 0, 1)) if len(img.shape) == 3 else img

            # normalize
            if key != self.label_key:
                img = (img - 128.0) / 128.0
            else:
                img[img > 0] = 1

            grid = []
            for x_idx in range(self.patch_per_side):
                for y_idx in range(self.patch_per_side):
                    x_start = x_idx * self.patch_size
                    x_end = x_start + self.patch_size
                    y_start = y_idx * self.patch_size
                    y_end = y_start + self.patch_size

                    if key != self.label_key:
                        grid.append(img[:, x_start:x_end, y_start:y_end])
                    else:
                        i = img[x_start:x_end, y_start:y_end]
                        grid.append(1 if np.sum(i) / i.size > 0.5 else 0)
            d[key] = np.array(grid)
        return d
