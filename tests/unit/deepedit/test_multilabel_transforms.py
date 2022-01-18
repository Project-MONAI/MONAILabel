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

from monailabel.deepedit.multilabel.transforms import (
    AddGuidanceSignalCustomd,
    DiscardAddGuidanced,
    SelectLabelsAbdomenDatasetd,
    SingleLabelSelectiond,
    SplitPredsLabeld,
    ToCheckTransformd,
)

IMAGE = np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]])
LABEL = np.array([[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]])
MULTIMODALITY_IMAGE = np.random.rand(5, 5, 5)
MULTI_LABEL = np.random.randint(0, 6, (5, 5))
PRED = np.random.randint(0, 6, (5, 5))
LABEL_NAMES = {
    "spleen": 1,
    "right kidney": 2,
    "background": 0,
}

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
    DATA_1,
    4,
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
    "image": IMAGE,
    "label": MULTI_LABEL,
    "guidance": {
        "spleen": np.array([[[1, 0, 2, 2], [-1, -1, -1, -1]]]),
        "right kidney": np.array([[[1, 0, 2, 2], [-1, -1, -1, -1]]]),
        "background": np.array([[[1, 0, 2, 2], [-1, -1, -1, -1]]]),
    },
    "discrepancy": {
        "spleen": np.array(
            [
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            ]
        ),
        "right kidney": np.array(
            [
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            ]
        ),
        "background": np.array(
            [
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            ]
        ),
    },
    "probability": 1.0,
    "label_names": LABEL_NAMES,
}

PosNegClickProbAddRandomGuidanceCustomd_TEST_CASE = [
    {"guidance": "guidance", "discrepancy": "discrepancy", "probability": "probability"},
    DATA_5,
    4,
]

DATA_6 = {
    "image": IMAGE,
    "label": MULTI_LABEL,
    "guidance": {
        "spleen": np.array([[[1, 0, 2, 2], [-1, -1, -1, -1]]]),
        "right kidney": np.array([[[1, 0, 2, 2], [-1, -1, -1, -1]]]),
        "background": np.array([[[1, 0, 2, 2], [-1, -1, -1, -1]]]),
    },
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "pred": PRED,
}

FindDiscrepancyRegionsCustomd_TEST_CASE = [
    {"discrepancy": "discrepancy"},
    DATA_6,
    (5, 5),
]

SelectLabelsAbdomenDatasetd_TEST_CASE = [
    {"label_names": LABEL_NAMES},
    DATA_6,
    len(LABEL_NAMES),
]

ADD_GUIDANCE_CUSTOM_TEST_CASE = [
    DATA_6,
    4,
]


DATA_7 = {
    "image": IMAGE,
    "label": MULTI_LABEL,
    "current_label": "spleen",
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "pred": PRED,
}

SingleLabelSelectiond_TEST_CASE = [
    {"label_names": ["spleen"]},
    DATA_7,
    "spleen",
]

DATA_8 = {
    "image": IMAGE,
    "label": MULTI_LABEL,
    "current_label": "spleen",
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "pred": PRED,
}

SplitPredsLabeld_TEST_CASE = [DATA_7]

ToCheckTransformd_TEST_CASE = [DATA_7, 6]

SingleModalityLabelSanityd_TEST_CASE = [DATA_7, (5, 5)]


DATA_9 = {
    "image": IMAGE,
    "label": MULTI_LABEL,
    "current_label": "spleen",
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "pred": PRED,
    "sids": {"spleen": [1, 2, 3]},
}

ADD_INITIAL_POINT_TEST_CASE_1 = [
    {"guidance": "guidance", "sids": "sids"},
    DATA_9,
    "[[[1, 0], [1, 2], [1, 0], [1, 0], [1, 2]]]",
]

# When checking tensor content use np.testing.assert_equal(result["image"], expected_values)


class TestDiscardAddGuidanced(unittest.TestCase):
    @parameterized.expand([DISCARD_ADD_GUIDANCE_TEST_CASE])
    def test_correct_results(self, input_data, expected_result):
        add_fn = DiscardAddGuidanced(keys="image", label_names=LABEL_NAMES)
        result = add_fn(input_data)
        np.testing.assert_equal(result["image"].shape[0], expected_result)


class TestSelectLabelsAbdomenDatasetd(unittest.TestCase):
    @parameterized.expand([SelectLabelsAbdomenDatasetd_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = SelectLabelsAbdomenDatasetd(keys="label", **arguments)
        result = add_fn(input_data)
        self.assertEqual(len(np.unique(result["label"])), expected_result)


class TestSingleLabelSelectiond(unittest.TestCase):
    @parameterized.expand([SingleLabelSelectiond_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = SingleLabelSelectiond(keys="label", **arguments)
        result = add_fn(input_data)
        self.assertEqual(result["current_label"], expected_result)


class TestAddGuidanceSignalCustomd(unittest.TestCase):
    @parameterized.expand([ADD_GUIDANCE_CUSTOM_TEST_CASE])
    def test_correct_results(self, input_data, expected_result):
        add_fn = AddGuidanceSignalCustomd(keys="image")
        result = add_fn(input_data)
        self.assertEqual(result["image"].shape[0], expected_result)


class TestSplitPredsLabeld(unittest.TestCase):
    @parameterized.expand([SplitPredsLabeld_TEST_CASE])
    def test_correct_results(self, input_data):
        add_fn = SplitPredsLabeld(keys="pred")
        result = add_fn(input_data)
        self.assertIsNotNone(result["pred_spleen"])


# Simple transform to debug other transforms
class TestToCheckTransformd(unittest.TestCase):
    @parameterized.expand([ToCheckTransformd_TEST_CASE])
    def test_correct_results(self, input_data, expected_result):
        add_fn = ToCheckTransformd(keys="label")
        result = add_fn(input_data)
        self.assertEqual(len(result), expected_result)


# # WORK IN PROGRESS
# class TestPosNegClickProbAddRandomGuidanceCustomd(unittest.TestCase):
#     @parameterized.expand([PosNegClickProbAddRandomGuidanceCustomd_TEST_CASE])
#     def test_correct_results(self, arguments, input_data, expected_result):
#         seed = 0
#         add_fn = PosNegClickProbAddRandomGuidanceCustomd(keys="NA", **arguments)
#         add_fn.set_random_state(seed)
#         result = add_fn(input_data)
#         self.assertGreaterEqual(len(result[arguments["guidance"]]["spleen"]), expected_result)
#
#
# class TestFindDiscrepancyRegionsCustomd(unittest.TestCase):
#     @parameterized.expand([FindDiscrepancyRegionsCustomd_TEST_CASE])
#     def test_correct_results(self, arguments, input_data, expected_result):
#         add_fn = FindDiscrepancyRegionsCustomd(keys="label", **arguments)
#         result = add_fn(input_data)
#         self.assertEqual(result["discrepancy"]["spleen"][0].shape, expected_result)
#
#
# class TestSingleModalityLabelSanityd(unittest.TestCase):
#     @parameterized.expand([SingleModalityLabelSanityd_TEST_CASE])
#     def test_correct_results(self, input_data, expected_result):
#         add_fn = SingleModalityLabelSanityd(keys="label")
#         result = add_fn(input_data)
#         self.assertEqual(result["label"].shape, expected_result)
#
#
# class TestAddInitialSeedPointd(unittest.TestCase):
#     @parameterized.expand([ADD_INITIAL_POINT_TEST_CASE_1])
#     def test_correct_results(self, arguments, input_data, expected_result):
#         seed = 0
#         add_fn = AddInitialSeedPointCustomd(keys='label', **arguments)
#         add_fn.set_random_state(seed)
#         result = add_fn(input_data)
#         self.assertEqual(result['guidance']['background'], expected_result)


if __name__ == "__main__":
    unittest.main()
