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

from monailabel.transform.post import BoundingBoxd, ExtremePointsd, LargestCCd, Restored

CCD_DATA = [
    {"keys": ("pred",)},
    {"pred": np.array([[[1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]])},
    np.array([[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]),
]

EXTREME_POINTS_DATA = [
    {"keys": "pred"},
    {"pred": np.array([[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])},
    [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
]

BB_DATA = [
    {"keys": "pred"},
    {"pred": np.array([[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]])},
    [[1, 1], [3, 3]],
]

RESTORED_DATA = [
    {"keys": "pred", "ref_image": "ref"},
    {
        "pred": np.array([[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]),
        "ref_meta_dict": {
            "spatial_shape": [1, 6, 6],
        },
    },
    (1, 6, 6),
]


class TestLargestCCd(unittest.TestCase):
    @parameterized.expand([CCD_DATA])
    def test_result(self, args, input_data, expected_output):
        res = LargestCCd(**args)(input_data)
        np.testing.assert_equal(res["pred"], expected_output)


class TestExtremePointsd(unittest.TestCase):
    @parameterized.expand([EXTREME_POINTS_DATA])
    def test_result(self, args, input_data, expected_data):
        res = ExtremePointsd(**args)(input_data)
        self.assertEqual(res["result"]["points"], expected_data)


class TestBoundingBoxd(unittest.TestCase):
    @parameterized.expand([BB_DATA])
    def test_result(self, args, input_data, expected_data):
        res = BoundingBoxd(**args)(input_data)
        self.assertEqual(res["result"]["bbox"], expected_data)


class TestRestored(unittest.TestCase):
    @parameterized.expand([RESTORED_DATA])
    def test_result(self, args, input_data, expected_shape):
        res = Restored(**args)(input_data)
        self.assertEqual(res["pred"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
