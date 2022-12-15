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

from monailabel.utils.others.generic import get_basename_no_ext


GETBUNDLE_CONFIG = [
    {
        "models": "spleen_ct_segmentation_v0.3.1",
    },
]

# class TestGetBundle(unittest.TestCase):
#     @parameterized.expand([GETBUNDLE_CONFIG])
#     def test_result(self, conf):
#         res = get_basename_no_ext()
#         print(res)
#         self.assertTrue(os.path.exists(res["spleen_ct_segmentation_v0.3.1"]))

if __name__ == "__main__":
    unittest.main()
