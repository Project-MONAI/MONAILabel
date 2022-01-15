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
import glob
import logging
import os

import numpy as np
from PIL import Image
from torchvision import transforms  # noqa

logger = logging.getLogger(__name__)


class PreProcess:
    def __init__(self):
        self._img_size = 4096
        self._patch_size = 512
        self._crop_size = 512
        self._normalize = True
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)

        if self._img_size % self._patch_size != 0:
            raise Exception(f"Image size / patch size != 0 : {self._img_size} / {self._patch_size}")

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        root_dir = "/local/sachi/Data/Pathology/Camelyon"
        self._images = sorted(glob.glob(f"{root_dir}/dataset/training/images/*.png"))
        self._labels = sorted(glob.glob(f"{root_dir}/dataset/training/labels/*.png"))

    def getitem(self, idx):
        image = self._images[idx]
        label = self._labels[idx]
        assert os.path.basename(image) == os.path.basename(label)

        img = Image.open(image)
        lab = Image.open(label)

        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        lab = lab.rotate(90 * num_rotate)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        lab = np.array(lab, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        # flatten the square grid
        img_flat = np.zeros((self._grid_size, 3, self._crop_size, self._crop_size), dtype=np.float32)
        lab_flat = np.zeros(self._grid_size, dtype=np.float32)

        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int((x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int((y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size

                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]
                lab_flat[idx] = lab[x_idx, y_idx]

                idx += 1

        return (img_flat, lab_flat)
