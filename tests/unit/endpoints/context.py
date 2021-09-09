# Copyright 2020 - 2021 MONAI Consortium
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
import logging
import os
import random
import sys
import unittest

from fastapi.testclient import TestClient

from monailabel.config import settings
from monailabel.main import app


def create_client(app_dir, studies):
    settings.MONAI_LABEL_APP_DIR = app_dir
    settings.MONAI_LABEL_STUDIES = studies
    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_APP_CONF = {"use_experiment_planner": "false", "tta_enabled": "false", "tta_samples": "1"}

    sys.path.append(settings.MONAI_LABEL_APP_DIR)
    sys.path.append(os.path.join(settings.MONAI_LABEL_APP_DIR, "lib"))
    for k, v in settings.dict().items():
        v = json.dumps(v) if isinstance(v, list) or isinstance(v, dict) else str(v)
        os.environ[k] = v
        logging.info(f"{k} => {v}")

    logs_dir = os.path.join(app_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    open(os.path.join(logs_dir, "app.log"), "a").close()

    return TestClient(app)


class BasicEndpointTestSuite(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "deepedit")
    studies = os.path.join(data_dir, "dataset", "local", "heart")
    rand_id = random.randint(0, 9999)

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = create_client(cls.app_dir, cls.studies)


class DICOMWebEndpointTestSuite(unittest.TestCase):
    client = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data", "dataset", "dicomweb")

    app_dir = os.path.join(base_dir, "sample-apps", "deepedit")
    studies = "http://faketesturl:8042/dicom-web"
    rand_id = random.randint(0, 9999)

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = create_client(cls.app_dir, cls.studies)
