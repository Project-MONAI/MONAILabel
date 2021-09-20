import os
import pathlib
import unittest

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


class TestWriter(unittest.TestCase):
    @parameterized.expand([WRITER_DATA])
    def test_nifti(self, args, input_data):
        args.update({"nibabel": True})
        output_file, data = Writer(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)

        file_ext = "".join(pathlib.Path(input_data["image_path"]).suffixes)
        self.assertIn(file_ext.lower(), [".nii", ".nii.gz"])

    @parameterized.expand([WRITER_DATA])
    def test_itk(self, args, input_data):
        args.update({"nibabel": False})
        output_file, data = Writer(**args)(input_data)
        self.assertEqual(os.path.exists(output_file), True)

        file_ext = "".join(pathlib.Path(input_data["image_path"]).suffixes)
        self.assertIn(file_ext.lower(), [".nii", ".nii.gz"])


if __name__ == "__main__":
    unittest.main()
