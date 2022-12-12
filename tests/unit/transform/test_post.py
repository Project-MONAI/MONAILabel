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

import os
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from monailabel.transform.post import (
    BoundingBoxd,
    DumpImagePrediction2Dd,
    ExtremePointsd,
    FindContoursd,
    LargestCCd,
    MergeAllPreds,
    RenameKeyd,
    Restored,
)

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

FINDCONTOURSD_DATA = [
    {"keys": "pred", "labels": "Other", "min_positive": 4, "min_poly_area": 1},
    {
        "pred": np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]),
    },
    [[[1, 2], [2, 1], [3, 2], [2, 3]], [[1, 1], [1, 3], [3, 3], [3, 1]]],
]

DUMPIMAGEPREDICTION2DD_DATA = [
    {
        "image": np.random.rand(1, 3, 5, 5),
        "pred": np.random.rand(1, 5, 5),
    },
]

METGEAllPREDS_DATA = [
    {"keys": ["pred", "pred_2"]},
    {
        "pred": np.array([[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]),
        "pred_2": np.array([[[1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]),
    },
    [[[1, 0, 0, 1], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]]],
]

RENAMEKEY_DATA = [
    {"source_key": "pred", "target_key": "pred_2"},
    {"pred": np.array([[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])},
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


class TestFindContoursd(unittest.TestCase):
    @parameterized.expand([FINDCONTOURSD_DATA])
    def test_result(self, args, input_data, expected_output):
        res = FindContoursd(**args)(input_data)
        self.assertEqual(res["result"]["annotation"]["elements"][0]["contours"], expected_output)


class TestDumpImagePrediction2Dd(unittest.TestCase):
    @parameterized.expand([DUMPIMAGEPREDICTION2DD_DATA])
    def test_saved_content(self, input_data):
        with tempfile.TemporaryDirectory() as tempdir:
            image_path = os.path.join(tempdir, "testimage.png")
            pred_path = os.path.join(tempdir, "testpred.png")
            _ = DumpImagePrediction2Dd(image_path=image_path, pred_path=pred_path, pred_only=False)(input_data)
            self.assertTrue(os.path.exists(image_path))
            self.assertTrue(os.path.exists(pred_path))


class TestMergeAllPreds(unittest.TestCase):
    @parameterized.expand([METGEAllPREDS_DATA])
    def test_merge_pred(self, args, input_data, expected_output):
        res = MergeAllPreds(**args)(input_data)
        self.assertEqual(res.tolist(), expected_output)


class TestRenameKeyd(unittest.TestCase):
    @parameterized.expand([RENAMEKEY_DATA])
    def test_rename_key(self, args, input_data):
        res = RenameKeyd(**args)(input_data)
        self.assertEqual(list(res.keys())[0], args["target_key"])


if __name__ == "__main__":
    unittest.main()
