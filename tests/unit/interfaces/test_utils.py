import unittest

import numpy as np
from monai.transforms.utility.array import Identity
from parameterized import parameterized

from monailabel.interfaces.utils.transform import run_transforms, shape_info

SHAPE_DATA = [
    {"pred": np.array([[[1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]).astype(np.float32)},
    ["pred"],
    "pred: (1, 3, 4)(float32)",
]

TRANSFORM_DATA = [{"pred": np.array([[[1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]).astype(np.float32)}, [Identity()]]


class TestTransformUtils(unittest.TestCase):
    @parameterized.expand([SHAPE_DATA])
    def test_shape_info(self, data, keys, expected_output):

        res = shape_info(data, keys)
        self.assertEqual(res, expected_output)

    @parameterized.expand([TRANSFORM_DATA])
    def test_run_transforms(self, data, callables):

        res = run_transforms(data, callables)
        np.testing.assert_equal(res, data)


if __name__ == "__main__":
    unittest.main()
