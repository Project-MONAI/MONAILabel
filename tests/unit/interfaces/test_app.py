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
import unittest

import torch
from parameterized import parameterized

from monailabel.config import settings
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.batch_infer import BatchInferImageType
from monailabel.interfaces.utils.app import app_instance


class TestApp(unittest.TestCase):
    app = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "radiology")
    studies = os.path.join(data_dir, "dataset", "local", "spleen")

    @classmethod
    def setUpClass(cls) -> None:
        settings.MONAI_LABEL_APP_DIR = cls.app_dir
        settings.MONAI_LABEL_STUDIES = cls.studies
        settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False

        if torch.cuda.is_available():
            cls.app: MONAILabelApp = app_instance(
                app_dir=cls.app_dir,
                studies=cls.studies,
                conf={
                    "preload": "true",
                    "models": "segmentation_spleen",
                },
            )

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_app_init(self):
        if not self.app:
            return
        self.app.on_init_complete()

    def test_cleanup_sessions(self):
        if not self.app:
            return
        self.app.cleanup_sessions()

    def test_async_batch_infer(self):
        if not self.app:
            return

        model = "segmentation_spleen"
        params = {"max_workers": 2}

        self.app.server_mode(True)
        self.app.async_batch_infer(model, BatchInferImageType.IMAGES_ALL, params)

        try:
            self.app.server_mode(False)
            self.app.async_batch_infer(model, BatchInferImageType.IMAGES_LABELED)
        except:
            pass

    def test_async_train(self):
        if not self.app:
            return

        model = "segmentation_spleen"
        params = {"max_epochs": 1}

        self.app.server_mode(True)
        self.app.async_training(model, params)

        try:
            self.app.server_mode(False)
            self.app.async_training(model, params)
        except:
            pass

    @parameterized.expand(["xnat", "dsa", ""])
    def test_init_datastores(self, r):
        if not self.app:
            return

        try:
            settings.MONAI_LABEL_DATASTORE = r
            self.app.init_remote_datastore()
        except:
            pass
        finally:
            settings.MONAI_LABEL_DATASTORE = ""


if __name__ == "__main__":
    unittest.main()
