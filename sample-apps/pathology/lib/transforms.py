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
from monai.transforms import CenterSpatialCrop, MapTransform, RandomizableTransform
from PIL import Image
from torchvision.transforms import ColorJitter

logger = logging.getLogger(__name__)


class LoadImagePatchd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(f"Type of {key} is: ")
            img = Image.open(d[key]) if isinstance(d[key], str) else d[key]
            img = np.array(img, dtype=np.uint8)
            d[key] = img
        return d


# You can write your transforms here... which can be used in your train/infer tasks
class ImageToNumpyd(MapTransform, RandomizableTransform):
    def __init__(
        self,
        keys: KeysCollection,
        jitter=True,
        flip=False,
        rotate=False,
        normalize=True,
        brightness=64.0 / 255.0,
        contrast=0.75,
        saturation=0.25,
        hue=0.04,
        label_key="label",
    ):
        super().__init__(keys)

        self.jitter = jitter
        self.flip = flip
        self.rotate = rotate
        self.normalize = normalize

        self.label_key = label_key
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data):
        d = dict(data)
        flip_right = self.R.uniform()
        num_rotate = self.R.randint(low=0, high=4)

        for key in self.keys:
            img = Image.open(d[key]) if isinstance(d[key], str) else d[key]

            # jitter (image only)
            if self.jitter and key != self.label_key:
                img = self.color_jitter(img)

            # flip
            if self.flip and flip_right > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # rotate
            if self.rotate:
                img = img.rotate(90 * num_rotate)

            # to numpy
            img = np.array(img, dtype=np.float32)
            img = img.transpose((2, 0, 1)) if len(img.shape) == 3 else img

            # normalize (image only)
            if self.normalize and key != self.label_key:
                img = (img - 128.0) / 128.0
            d[key] = img
        return d


class LabelToChanneld(MapTransform):
    def __init__(self, keys: KeysCollection, labels):
        super().__init__(keys)
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask = d[key]
            img = np.zeros((len(self.labels), mask.shape[0], mask.shape[1]))

            count = 0
            for idx in self.labels:
                img[count, mask == idx] = 1
                count += 1
            d[key] = img
        return d


class ClipBorderd(MapTransform):
    def __init__(self, keys: KeysCollection, border=2):
        super().__init__(keys)
        self.border = border

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            roi_size = (img.shape[-2] - self.border * 2, img.shape[-1] - self.border * 2)
            crop = CenterSpatialCrop(roi_size=roi_size)
            d[key] = crop(img)
        return d
