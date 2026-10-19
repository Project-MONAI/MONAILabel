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

import builtins
import importlib
import sys
import unittest
from unittest import mock

import numpy as np
from monai.utils import OptionalImportError

import monailabel.scribbles.utils as scribbles_utils


class TestOptionalNumpymaxflow(unittest.TestCase):
    """Scribbles must import without numpymaxflow and fail only when GraphCut is actually used."""

    def setUp(self):
        self._modules = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "numpymaxflow" or name.startswith("numpymaxflow.") or name.startswith("monailabel.scribbles")
        }
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "numpymaxflow" or name.startswith("numpymaxflow."):
                raise ImportError("No module named 'numpymaxflow'")
            return real_import(name, *args, **kwargs)

        self._patch = mock.patch("builtins.__import__", side_effect=blocked_import)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        for name in list(sys.modules):
            if name.startswith("monailabel.scribbles"):
                sys.modules.pop(name)
        sys.modules.update(self._modules)

    def test_scribbles_modules_import_without_numpymaxflow(self):
        utils = importlib.import_module("monailabel.scribbles.utils")
        self.assertFalse(utils.has_numpymaxflow)
        importlib.import_module("monailabel.scribbles.transforms")
        importlib.import_module("monailabel.scribbles.infer")

    def test_maxflow_raises_when_numpymaxflow_missing(self):
        utils = importlib.import_module("monailabel.scribbles.utils")
        image = np.zeros((1, 4, 4), dtype=np.float32)
        prob = np.full((2, 4, 4), 0.5, dtype=np.float32)
        with self.assertRaises(OptionalImportError) as ctx:
            utils.maxflow(image, prob)
        self.assertEqual(str(ctx.exception), utils.missing_numpymaxflow_message())

    def test_message_on_supported_python_recommends_the_extra(self):
        utils = importlib.import_module("monailabel.scribbles.utils")
        message = utils.missing_numpymaxflow_message((3, 12))
        self.assertIn("monailabel[scribbles]", message)

    def test_message_on_unsupported_python_says_graphcut_is_unavailable(self):
        utils = importlib.import_module("monailabel.scribbles.utils")
        message = utils.missing_numpymaxflow_message((3, 13))
        self.assertIn("unavailable on Python 3.13", message)
        self.assertNotIn("pip install", message)

    def test_maxflow_on_unsupported_python_says_graphcut_is_unavailable(self):
        utils = importlib.import_module("monailabel.scribbles.utils")
        image = np.zeros((1, 4, 4), dtype=np.float32)
        prob = np.full((2, 4, 4), 0.5, dtype=np.float32)
        with mock.patch.object(utils.sys, "version_info", (3, 13, 0, "final", 0)):
            with self.assertRaises(OptionalImportError) as ctx:
                utils.maxflow(image, prob)
        self.assertIn("unavailable on Python 3.13", str(ctx.exception))
        self.assertNotIn("pip install", str(ctx.exception))

    def test_non_graphcut_helpers_still_work(self):
        utils = importlib.import_module("monailabel.scribbles.utils")
        self.assertGreater(utils.get_eps(np.zeros(1, dtype=np.float32)), 0)


@unittest.skipUnless(scribbles_utils.has_numpymaxflow, "numpymaxflow is not installed")
class TestNumpymaxflowPresent(unittest.TestCase):
    def test_maxflow_runs_when_installed(self):
        image = np.random.rand(1, 8, 8).astype(np.float32)
        prob = np.random.rand(2, 8, 8).astype(np.float32)
        prob = prob / prob.sum(axis=0, keepdims=True)
        label = scribbles_utils.maxflow(image, prob)
        self.assertEqual(label.shape[-2:], (8, 8))


if __name__ == "__main__":
    unittest.main()
