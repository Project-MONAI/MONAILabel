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
from parameterized import parameterized

from monailabel.deepedit.transforms import (
    AddRandomGuidanced,
    CropGuidanceForegroundd,
    DiscardAddGuidanced,
    PosNegClickProbAddRandomGuidanced,
    ResizeGuidanceCustomd,
    ResizeGuidanceWithPadOrCropd,
)

IMAGE = np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]])
LABEL = np.array([[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]])
MULTIMODALITY_IMAGE = np.random.rand(5, 5, 5)
MULTI_LABEL = np.random.randint(0, 6, (5, 5))
PRED = np.random.randint(0, 6, (5, 5))

DATA_1 = {
    "image": IMAGE,
    "label": LABEL,
    "image_meta_dict": {
        "dim": np.array(IMAGE.shape),
        "pixdim": [1, 1, 1, 5, 1, 1, 1, 1],
        "filename_or_obj": "IMAGE_NAME",
    },
    "label_meta_dict": {},
    "foreground": [0, 0, 0],
    "background": [0, 0, 0],
}

DISCARD_ADD_GUIDANCE_TEST_CASE = [
    {"image": IMAGE, "label": LABEL},
    DATA_1,
    # Image
    [
        [
            [
                [1.0, 0.0, 2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [2.0, 2.0, 3.0, 2.0, 2.0],
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        ],
        # Positive clicks in zeros
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        # Negative clicks in zeros
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
    ],
]

DATA_2 = {
    "image": IMAGE,
    "label": LABEL,
    "guidance": np.array([[[1, 0, 2, 2]], [[-1, -1, -1, -1]]]),
    "discrepancy": np.array(
        [
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
        ]
    ),
    "probability": 1.0,
}

CLICK_RATIO_ADD_RANDOM_GUIDANCE_TEST_CASE_1 = [
    {"guidance": "guidance", "discrepancy": "discrepancy", "probability": "probability"},
    DATA_2,
    "[[[1, 0, 2, 2], [-1, -1, -1, -1]], [[-1, -1, -1, -1], [1, 0, 2, 1]]]",
]

ADD_RANDOM_GUIDANCE_TEST_CASE_1 = [
    {"guidance": "guidance", "discrepancy": "discrepancy", "probability": "probability"},
    DATA_2,
    "[[[1, 0, 2, 2], [1, 0, 1, 3]], [[-1, -1, -1, -1], [-1, -1, -1, -1]]]",
]

DATA_3 = {
    "image": np.arange(1000).reshape((1, 5, 10, 20)),
    "image_meta_dict": {
        "foreground_cropped_shape": (1, 10, 20, 40),
        "dim": [3, 512, 512, 128],
        "spatial_shape": [512, 512, 128],
    },
    "guidance": [[[6, 10, 14], [8, 10, 14]], [[8, 10, 16]]],
    "foreground": [[10, 14, 6], [10, 14, 8]],
    "background": [[10, 16, 8]],
}

RESIZE_GUIDANCE_TEST_CASE_1 = [
    {"ref_image": "image", "guidance": "guidance"},
    DATA_3,
    [[[0, 0, 0], [0, 0, 1]], [[0, 0, 1]]],
]

DATA_4 = {
    "image": MULTIMODALITY_IMAGE,
    "label": MULTI_LABEL,
    "image_meta_dict": {
        "dim": [1, 1, 1, 5, 2, 1, 1, 1],
        "pixdim": [1, 1, 1, 5, 2, 1, 1, 1],
        "filename_or_obj": "IMAGE_NAME",
    },
    "label_meta_dict": {
        "dim": [1, 1, 1, 5, 2, 1, 1, 1],
        "pixdim": [1, 1, 1, 5, 2, 1, 1, 1],
        "filename_or_obj": "LABEL_NAME",
    },
}

DATA_5 = {
    "image": np.arange(8000).reshape((1, 20, 20, 20)),
    "guidance": [[[7, 8, 9], [10, 10, 10]], [[12, 15, 9]]],
    "mask": np.expand_dims(np.pad(np.ones((16, 16, 16)), pad_width=2), axis=0),
}

CROP_GUIDANCE_FOREGROUND_TEST_CASE_1 = [
    {"ref_image": "image", "guidance": "guidance", "source_key": "mask"},
    DATA_5,
    [[[5, 6, 7], [8, 8, 8]], [[10, 13, 7]]],
]

CROP_GUIDANCE_FOREGROUND_TEST_CASE_2 = [
    {"ref_image": "image", "guidance": "guidance", "source_key": "mask"},
    DATA_5,
    [[[6, 7, 8], [9, 9, 9]], [[11, 14, 8]]],
]

DATA_6 = {
    "image": np.arange(8000).reshape((1, 20, 20, 20)),
    "guidance": [[[7, 8, 9], [10, 10, 10]], [[12, 15, 9]]],
}

RESIZE_GUIDANCE_WITH_PAD_OR_CROP_TEST_CASE_1 = [
    {"ref_image": "image", "guidance": "guidance"},
    DATA_6,
    [[[5, 10, 7], [8, 12, 8]], [[10, 17, 7]]],
]

# When checking tensor content use np.testing.assert_equal(result["image"], expected_values)


class TestDiscardAddGuidanced(unittest.TestCase):
    @parameterized.expand([DISCARD_ADD_GUIDANCE_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = DiscardAddGuidanced(arguments)
        result = add_fn(input_data)
        np.testing.assert_equal(result["image"], expected_result)


class TestClickRatioAddRandomGuidanced(unittest.TestCase):
    @parameterized.expand([CLICK_RATIO_ADD_RANDOM_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = PosNegClickProbAddRandomGuidanced(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestAddRandomGuidanced(unittest.TestCase):
    @parameterized.expand([ADD_RANDOM_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddRandomGuidanced(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestResizeGuidanced(unittest.TestCase):
    @parameterized.expand([RESIZE_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = ResizeGuidanceCustomd(**arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class Test1CropGuidanceForegroundd(unittest.TestCase):
    @parameterized.expand([CROP_GUIDANCE_FOREGROUND_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = CropGuidanceForegroundd(**arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class Test2CropGuidanceForegroundd(unittest.TestCase):
    @parameterized.expand([CROP_GUIDANCE_FOREGROUND_TEST_CASE_2])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = CropGuidanceForegroundd(margin=(1, 1, 1), **arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestResizeGuidanceWithPadOrCropd(unittest.TestCase):
    @parameterized.expand([RESIZE_GUIDANCE_WITH_PAD_OR_CROP_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = ResizeGuidanceWithPadOrCropd(spatial_size=(16, 24, 16), **arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


if __name__ == "__main__":
    unittest.main()
