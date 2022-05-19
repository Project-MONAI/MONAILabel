import os
import pathlib
import unittest

import nrrd
import numpy as np
from parameterized import parameterized

from monailabel.transform.writer import Writer

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

        space_directions_expected = np.array(
            [[np.nan, np.nan, np.nan], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertTrue(np.array_equal(header["space directions"], space_directions_expected, equal_nan=True))

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


if __name__ == "__main__":
    unittest.main()
