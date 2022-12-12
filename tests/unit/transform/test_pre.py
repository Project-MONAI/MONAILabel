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

import unittest

import numpy as np
from parameterized import parameterized

from monailabel.transform.pre import LoadImageExd, LoadImageTensord, NormalizeLabeld

LOADIMAGETENSOR_DATA = [
    {"keys": "image"},
    {
        "image": np.random.rand(64, 64, 64),
    },
    (64, 64, 64),
]

NORMALIZELABELD_DATA = [
    {"keys": "label"},
    {
        "label": np.array([[[0, 0, 0, 0], [0, 5, 4, 0], [0, 3, 2, 0], [0, 1, 0, 0]]]),
    },
    [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]]],
]


class TestLoadImageTensord(unittest.TestCase):
    @parameterized.expand([LOADIMAGETENSOR_DATA])
    def test_load_image_shape(self, args, input_data, expected_shape):
        res = LoadImageTensord(**args)(input_data)
        self.assertTupleEqual(res["image"].shape, expected_shape)


class TestLoadImageExd(unittest.TestCase):
    @parameterized.expand([LOADIMAGETENSOR_DATA])
    def test_load_imageEx_shape(self, args, input_data, expected_shape):
        res = LoadImageExd(**args)(input_data)
        self.assertTupleEqual(res["image"].shape, expected_shape)


class TestNormalizeLabeld(unittest.TestCase):
    @parameterized.expand([NORMALIZELABELD_DATA])
    def test_result(self, args, input_data, expected_data):
        res = NormalizeLabeld(**args)(input_data)
        self.assertEqual(res["label"].tolist(), expected_data)


if __name__ == "__main__":
    unittest.main()
