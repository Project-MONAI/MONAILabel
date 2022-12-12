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

import json
import os
import pathlib
import unittest

import nrrd
import numpy as np
import torch
from parameterized import parameterized

from monailabel.transform.writer import DetectionWriter, PolygonWriter, Writer

WRITER_DATA = [
    {"label": "pred"},
    {
        "pred": np.array([[[1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]).astype(np.float32),
        "pred_meta_dict": {
            "affine": np.identity(4),
        },
        "image_path": "fakepath.nii",
    },
]

CHANNELS = 2
WIDTH = 15
HEIGHT = 10
MULTI_CHANNEL_DATA = np.zeros((CHANNELS, WIDTH, HEIGHT, 1))

COLOR_MAP = {
    # according to getLabelColor() [https://github.com/Project-MONAI/MONAILabel/blob/6cc72c542c9bc6c5181af89550e7e397537d74e3/plugins/slicer/MONAILabel/MONAILabel.py#L1485] # noqa
    "lung": [128 / 255, 174 / 255, 128 / 255],  # green
    "heart": [206 / 255, 110 / 255, 84 / 255],  # red
}


POLYGONWRITER_DATA = [
    {"label": "pred"},
    {
        "pred": np.array([[[1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]).astype(np.float32),
        "pred_meta_dict": {
            "affine": np.identity(4),
        },
        "image_path": "fakepath.nii",
        "output": "asap",
        "result": {
            "annotation": {
                "location": (2263, 5298),
                "size": (55, 44),
                "elements": [
                    {
                        "label": "Inflammatory",
                        "contours": [
                            [
                                [2299, 5298],
                                [2299, 5301],
                                [2302, 5301],
                                [2303, 5302],
                                [2304, 5302],
                                [2308, 5306],
                                [2308, 5307],
                                [2309, 5307],
                                [2310, 5308],
                                [2311, 5307],
                                [2314, 5307],
                                [2315, 5306],
                                [2316, 5307],
                                [2316, 5306],
                                [2317, 5305],
                                [2317, 5298],
                            ]
                        ],
                    }
                ],
                "labels": {"Inflammatory": (255, 255, 0), "Spindle-Shaped": (0, 255, 0)},
            }
        },
    },
]


DETECTION_DATA = [
    {"pred_box_key": "box", "pred_label_key": "label"},
    {
        "pred": np.array([[[1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]).astype(np.float32),
        "pred_meta_dict": {
            "affine": np.identity(4),
        },
        "image_path": "fakepath.nii",
        "box": torch.tensor(
            [
                [37.5524, -153.9518, -230.8054, 8.1992, 8.1875, 8.2574],
                [-112.3948, -160.7483, -192.0955, 4.8497, 4.7852, 4.8144],
                [-8.1750, -187.5624, -48.9165, 7.0694, 7.0703, 6.5331],
                [-98.9357, -225.9962, -176.4017, 4.5932, 4.6513, 4.4018],
                [-95.4652, -132.0942, -159.3038, 4.4789, 4.3853, 4.5687],
                [-69.8407, -195.1670, -102.0295, 4.2224, 4.2135, 4.2672],
                [-46.4637, -170.2633, -75.5516, 16.6934, 16.4776, 16.7486],
                [-61.1737, -222.7262, -130.8002, 8.5873, 8.4749, 8.4555],
            ]
        ),
        "label": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
    },
]


class TestWriter(unittest.TestCase):
    @parameterized.expand([WRITER_DATA])
    def test_nifti(self, args, input_data):
        args.update({"nibabel": True})
        output_file, data = Writer(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)

        file_ext = "".join(pathlib.Path(input_data["image_path"]).suffixes)
        self.assertIn(file_ext.lower(), [".nii", ".nii.gz"])

    @parameterized.expand([WRITER_DATA])
    def test_seg_nrrd(self, args, input_data):
        args.update({"nibabel": False})
        input_data["pred"] = MULTI_CHANNEL_DATA
        input_data["result_extension"] = ".seg.nrrd"
        input_data["labels"] = ["heart", "lung"]
        input_data["color_map"] = COLOR_MAP

        output_file, data = Writer(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)
        arr_full, header = nrrd.read(output_file)

        self.assertEqual(arr_full.shape, (CHANNELS, WIDTH, HEIGHT, 1))

        # TODO:: This depends on later versions of numpy (fix the input/output comparisons for nans)
        space_directions_expected = np.array(
            [[np.nan, np.nan, np.nan], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        # self.assertTrue(np.array_equal(header["space directions"], space_directions_expected, equal_nan=True))

        self.assertEqual(header["kinds"], ["list", "domain", "domain", "domain"])
        self.assertEqual(header["Segment1_ID"], "lung")
        self.assertEqual(header["Segment1_Color"], " ".join(map(str, COLOR_MAP["lung"])))

        file_ext = "".join(pathlib.Path(output_file).suffixes)
        self.assertIn(file_ext.lower(), [".seg.nrrd"])

    @parameterized.expand([WRITER_DATA])
    def test_itk(self, args, input_data):
        args.update({"nibabel": False})
        output_file, data = Writer(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)

        file_ext = "".join(pathlib.Path(input_data["image_path"]).suffixes)
        self.assertIn(file_ext.lower(), [".nii", ".nii.gz"])


class TestPolygonWriter(unittest.TestCase):
    @parameterized.expand([POLYGONWRITER_DATA])
    def test_asap_result(self, args, input_data):
        output_file, data = PolygonWriter(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)
        file_ext = "".join(pathlib.Path(input_data["image_path"]).suffixes)
        self.assertIn(file_ext.lower(), [".nii", ".nii.gz"])
        file_ext = "".join(pathlib.Path(output_file).suffixes)
        self.assertIn(file_ext.lower(), [".xml"])

    @parameterized.expand([POLYGONWRITER_DATA])
    def test_dsa_result(self, args, input_data):
        input_data.update({"output": "dsa"})
        output_file, data = PolygonWriter(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)
        file_ext = "".join(pathlib.Path(input_data["image_path"]).suffixes)
        self.assertIn(file_ext.lower(), [".nii", ".nii.gz"])
        file_ext = "".join(pathlib.Path(output_file).suffixes)
        self.assertIn(file_ext.lower(), [".json"])

        with open(output_file) as f:
            data = json.load(f)
            self.assertEqual(len(data["elements"][0]["points"]), 16)
            self.assertEqual(data["elements"][0]["label"]["value"], "Inflammatory")


class TestDetectionWriter(unittest.TestCase):
    @parameterized.expand([DETECTION_DATA])
    def test_slicer_result(self, args, input_data):
        output_file, data = DetectionWriter(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)
        file_ext = "".join(pathlib.Path(input_data["image_path"]).suffixes)
        self.assertIn(file_ext.lower(), [".nii", ".nii.gz"])
        file_ext = "".join(pathlib.Path(output_file).suffixes)
        self.assertIn(file_ext.lower(), [".json"])
        with open(output_file) as f:
            data = json.load(f)
            self.assertEqual(
                data["markups"][0]["center"], [37.552398681640625, -153.95179748535156, -230.80540466308594]
            )
            self.assertEqual(data["markups"][0]["size"], [8.199199676513672, 8.1875, 8.257399559020996])


if __name__ == "__main__":
    unittest.main()
