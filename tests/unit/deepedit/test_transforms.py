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

import unittest

import numpy as np

from monailabel.deepedit.transforms import DiscardAddGuidanced, ResizeGuidanceCustomd

IMAGE = np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]])
LABEL = np.array([[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]])
BATCH_IMAGE = np.array([[[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]]])
BATCH_LABEL = np.array([[[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]]])

DATA_1 = {
    "image": IMAGE,
    "label": LABEL,
    "image_meta_dict": {"dim": IMAGE.shape},
    "label_meta_dict": {},
    "foreground": [0, 0, 0],
    "background": [0, 0, 0],
}


class MyTestCase(unittest.TestCase):
    def test_t1(self):
        DiscardAddGuidanced()(DATA_1)

    def test_t2(self):
        ResizeGuidanceCustomd("label", "image")(DATA_1)


if __name__ == "__main__":
    unittest.main()
