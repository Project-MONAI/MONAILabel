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
from argparse import Namespace
from unittest.mock import patch


class MockAppInstance:
    def info(self):
        return {"name": "myapp"}

    def infer(self, request, datastore=None):
        return {"xyz": "xyz.nii.gz"}

    def train(self, request, datastore=None):
        return {"task": "1234"}

    def batch_infer(self, request, datastore=None):
        return {"result": "xyz"}

    def scoring(self, request, datastore=None):
        return {"result": "abc"}


def mock_app_instance(app_dir, studies):
    return MockAppInstance()


@patch("monailabel.interfaces.utils.app.app_instance", new=mock_app_instance)
class MyTestCase(unittest.TestCase):
    def test_info(self):
        from monailabel.interfaces.test import test_info

        test_info(Namespace(app="dummy", studies="dymmy", device="d"))

    def test_infer(self):
        from monailabel.interfaces.test import test_infer

        test_infer(
            Namespace(app="dummy", studies="dymmy", device="d", model="abc", input="i", output="o", runs=1, params="{}")
        )

    def test_train(self):
        from monailabel.interfaces.test import test_train

        test_train(Namespace(app="dummy", studies="dymmy", device="d", name="x", epochs=1, amp=True))


if __name__ == "__main__":
    unittest.main()
