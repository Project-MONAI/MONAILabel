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
import shutil
import tempfile
import unittest

from monailabel.utils.others.generic import path_to_sqlite_uri


class TestDefaultTrackingUri(unittest.TestCase):
    def setUp(self):
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="monailabel_tracking_"))

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_path_to_sqlite_uri(self):
        self.assertEqual(path_to_sqlite_uri("/data/model/mlruns.db"), "sqlite:////data/model/mlruns.db")
        relative = path_to_sqlite_uri("mlruns.db")
        self.assertTrue(relative.startswith("sqlite:///"))
        self.assertTrue(relative.endswith("/mlruns.db"))

    def test_default_uri_is_accepted_by_the_installed_mlflow_handler(self):
        """The default tracking URI must work with the MONAI/MLflow versions a fresh install resolves to."""
        from monai.handlers import MLFlowHandler

        uri = path_to_sqlite_uri(os.path.join(self.base, "output", "mlruns.db"))
        os.makedirs(os.path.join(self.base, "output"))
        handler = MLFlowHandler(tracking_uri=uri, experiment_name="monailabel_test", close_on_complete=True)
        self.assertEqual(handler.tracking_uri, uri)


if __name__ == "__main__":
    unittest.main()
