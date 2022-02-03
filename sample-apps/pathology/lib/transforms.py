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
        jitter=True,
        flip=True,
        rotate=True,
        normalize=True,
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
        logger.info(f"Keys: {self.keys}")
        for k, v in d.items():
            logger.info(f"{k} => {v}")

        for key in self.keys:
            logger.info(f"Open Image: {d[key]}")
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

            # normalize
            if key != self.label_key:
                if self.normalize:
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
            d[key] = np.array(grid, dtype=np.float32)
        return d


class ImageToGridBatchd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        image_size,
        patch_size,
        normalize=True,
    ):
        super().__init__(keys)

        if image_size % patch_size != 0:
            raise Exception("Image size / patch size != 0 : {} / {}".format(image_size, patch_size))

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_per_side = self.image_size // self.patch_size
        self.grid_size = self.patch_per_side * self.patch_per_side

        self.normalize = normalize

    def __call__(self, data):
        d = dict(data)
        for k, v in d.items():
            logger.info(f"{k} => {v}")

        for key in self.keys:
            batch = []
            for i in d[key]:
                img = Image.open(i) if isinstance(i, str) else i

                # to numpy
                img = np.array(img, dtype=np.float32)
                img = img.transpose((2, 0, 1)) if len(img.shape) == 3 else img

                # normalize
                if self.normalize:
                    img = (img - 128.0) / 128.0

                grid = []
                for x_idx in range(self.patch_per_side):
                    for y_idx in range(self.patch_per_side):
                        x_start = x_idx * self.patch_size
                        x_end = x_start + self.patch_size
                        y_start = y_idx * self.patch_size
                        y_end = y_start + self.patch_size

                        grid.append(img[:, x_start:x_end, y_start:y_end])
                batch.append(grid)
            d[key] = np.array(batch, dtype=np.float32)
        return d


class GridToLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        image_size,
        patch_size,
    ):
        super().__init__(keys)

        if image_size % patch_size != 0:
            raise Exception("Image size / patch size != 0 : {} / {}".format(image_size, patch_size))

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_per_side = self.image_size // self.patch_size
        self.grid_size = self.patch_per_side * self.patch_per_side

    def split(self, array, nrows, ncols):
        _, h = array.shape
        return array.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            probs = d[key]
            logger.info(f"{key}: {probs.shape} ({probs.dtype})")

            probs = probs.sigmoid().cpu().data.numpy()
            probs = np.where(probs > 0.5, 255, 0).astype(np.uint8)
            probs = probs[np.newaxis] if len(probs.shape) == 1 else probs

            label = np.zeros(
                (probs.shape[0], self.patch_size * self.patch_per_side, self.patch_size * self.patch_per_side),
                dtype=probs.dtype,
            )

            for batch_idx in range(probs.shape[0]):
                count = 0
                # partitions = self.split(np.reshape(probs[batch_idx], (16, 16)), 4, 4)
                # avg = [np.average(p) for p in partitions]

                for x_idx in range(self.patch_per_side):
                    for y_idx in range(self.patch_per_side):
                        x_start = x_idx * self.patch_size
                        x_end = x_start + self.patch_size
                        y_start = y_idx * self.patch_size
                        y_end = y_start + self.patch_size

                        label[batch_idx][x_start:x_end, y_start:y_end] = probs[batch_idx][count]

                        # normalize to grid/partition level
                        # p_index = 4 * (x_idx // 4) + (y_idx // 4)
                        # logger.debug(f"Index: {x_idx},{y_idx} => count:{count} => partition: {p_index} => {avg}")
                        # label[batch_idx][x_start:x_end, y_start:y_end] = 1 if avg[p_index] >= 0.5 else 0
                        count += 1

            logger.info(f"{key}: {label.shape} ({label.dtype})")
            d[key] = label
        return d
