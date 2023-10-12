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
from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirstd
from monai.utils import min_version, optional_import, set_determinism
from monai.utils.enums import PostFix
from parameterized import parameterized

from monailabel.deepedit.transforms import (
    AddClickGuidanced,
    AddGuidanceSignald,
    AddInitialSeedPointd,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropForegroundd,
    SpatialCropGuidanced,
)

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

set_determinism(seed=0)
IMAGE = np.random.randint(0, 256, size=(1, 10, 10, 10))
IMAGE2 = np.random.randint(0, 256, size=(10, 10, 10))

LABEL = np.random.randint(0, 2, size=(10, 10, 10))
LABEL_NAMES = {"spleen": 1, "background": 0}

set_determinism(None)

DATA_1 = {
    "image": IMAGE,
    "label": LABEL,
    "spleen": [1, 0, 2, 2],
    "background": [-1, -1, -1, -1],
    "probability": 1.0,
    "label_names": LABEL_NAMES,
}

DATA_2 = {
    "image": IMAGE,
    "label": LABEL,
    "spleen": [1, 0, 2, 2],
    "background": [-1, -1, -1, -1],
    "guidance": [[1, 0, 2, 2], [-1, -1, -1, -1]],
    "probability": 1.0,
    "label_names": LABEL_NAMES,
}

DATA_3 = {
    "image": IMAGE,
    "label": LABEL,
    "spleen": [1, 0, 2, 2],
    "background": [-1, -1, -1, -1],
    "guidance": [[[1, 0, 2, 2]], [[-1, -1, -1, -1]]],
    "probability": 1.0,
    "label_names": LABEL_NAMES,
}

DATA_4 = {
    "image": IMAGE2,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[...].shape},
    PostFix.meta("label"): {},
    "spleen": [1, 0, 2, 2],
    "background": [-1, -1, -1, -1],
    "guidance": [[[1, 0, 7]], [[1, 5, 3]]],
    "probability": 1.0,
    "label_names": LABEL_NAMES,
}

DATA_5 = {
    "image": MetaTensor(
        IMAGE2,
        meta={
            "dim": IMAGE.shape,
            "spatial_shape": IMAGE[...].shape,
            "spleen": [1, 0, 2, 2],
            "background": [-1, -1, -1, -1],
            "guidance": [[[1, 0, 7]], [[1, 5, 3]]],
            "probability": 1.0,
            "label_names": LABEL_NAMES,
            "foreground_start_coord": [0, 0, 0],
            "foreground_end_coord": [12, 12, 12],
            "foreground_original_shape": [1, 10, 10, 10],
            "foreground_cropped_shape": [1, 10, 10, 10],
        },
    ),
    "label": MetaTensor(LABEL),
    PostFix.meta("image"): {},
    PostFix.meta("label"): {},
}

AddCLICKGUIDENCD_DATA = [
    {"keys": ["spleen", "background"], "guidance": "guidance"},  # arguments
    DATA_1,  # input_data
    [1, 0, 2, 2],  # expected_result
]

ADDINITIALSEEDOINTD_DATA = [
    {"keys": "guidance", "label": "label"},  # arguments
    DATA_2,  # input_data
]

ADD_GUIDANCE_CUSTOM_TEST_CASE = [
    {"keys": "image", "guidance": "guidance"},  # arguments
    DATA_3,  # input_data
    3,  # expected_result
]

SPATIALCROPFOREGROUNDD_DATA = [
    {"keys": ["image", "label"], "source_key": "label", "spatial_size": (12, 12, 12)},  # arguments
    DATA_4,  # input_data
    [1, 10, 10, 10],  # expected_result
]

SPATIALCROPGUIDANCED_DATA = [
    {"keys": ["image", "label"], "guidance": "guidance", "spatial_size": (12, 12, 12), "margin": 0},  # arguments
    DATA_4,  # input_data
    [1, 10, 10, 10],  # expected_result
]

RESIZEGUIDANCE_DATA = [
    {"keys": "guidance", "ref_image": "image"},  # arguments
    DATA_4,  # input_data
    [[[1, 0, 7]], [[1, 5, 3]]],  # expected_result
]

RESTORELABELD_DATA = [
    {
        "keys": "label",
        "ref_image": "image",
        "mode": "nearest",
        "start_coord_key": "foreground_start_coord",
        "end_coord_key": "foreground_end_coord",
        "original_shape_key": "foreground_original_shape",
        "cropped_shape_key": "foreground_cropped_shape",
    },  # arguments
    DATA_5,  # input_data
    (10, 10, 10),  # expected_result
]


class TestAddClickGuidanced(unittest.TestCase):
    @parameterized.expand([AddCLICKGUIDENCD_DATA])
    def test_addclickguidanced(self, args, input_data, expected_result):
        add_fn = AddClickGuidanced(**args)
        result = add_fn(input_data)
        self.assertEqual(result[args["guidance"]][0], expected_result)


class TestAddInitialSeedPointd(unittest.TestCase):
    @parameterized.expand([ADDINITIALSEEDOINTD_DATA])
    def test_AddInitialSeedPointd(self, args, input_data):
        seed = 0
        add_fn = AddInitialSeedPointd(**args)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        assert result["guidance"] is not None


class TestAddGuidanceSignald(unittest.TestCase):
    @parameterized.expand([ADD_GUIDANCE_CUSTOM_TEST_CASE])
    def test_addguidancesignald(self, arguments, input_data, expected_result):
        add_fn = AddGuidanceSignald(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result["image"].shape[0], expected_result)


class TestSpatialCropForegroundd(unittest.TestCase):
    @parameterized.expand([SPATIALCROPFOREGROUNDD_DATA])
    def test_spatialcropforegroundd(self, arguments, input_data, expected_result):
        add_ch = EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel")(input_data)
        add_fn = SpatialCropForegroundd(**arguments)
        result = add_fn(add_ch)["label"].shape
        self.assertEqual(list(result), expected_result)


class TestSpatialCropGuidanced(unittest.TestCase):
    @parameterized.expand([SPATIALCROPGUIDANCED_DATA])
    def test_spatialcropguidanced(self, arguments, input_data, expected_result):
        add_ch = EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel")(input_data)
        add_fn = SpatialCropGuidanced(**arguments)
        result = add_fn(add_ch)["label"].shape
        self.assertEqual(list(result), expected_result)


class TestResizeGuidanced(unittest.TestCase):
    @parameterized.expand([RESIZEGUIDANCE_DATA])
    def test_resizeguidanced(self, arguments, input_data, expected_result):
        add_ch = EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel")(input_data)
        add_fn = ResizeGuidanced(**arguments)
        result = add_fn(add_ch)
        self.assertEqual(result["guidance"], expected_result)


class TestRestoreLabeld(unittest.TestCase):
    @parameterized.expand([RESTORELABELD_DATA])
    def test_restorelabeld(self, arguments, input_data, expected_result):
        add_ch = EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel")(input_data)
        add_fn = RestoreLabeld(**arguments)
        result = add_fn(add_ch)
        self.assertEqual(result["label"].shape, expected_result)


if __name__ == "__main__":
    unittest.main()
