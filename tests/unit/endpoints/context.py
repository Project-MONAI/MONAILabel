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
import json
import logging
import os
import random
import sys
import unittest

from fastapi.testclient import TestClient

from monailabel.config import settings


def create_client(app_dir, studies, data_dir, conf=None):
    app_conf = {
        "heuristic_planner": "false",
        "server_mode": "true",
        "auto_update_scoring": "false",
        "debug": "true",
        "models": "deepedit",
    }
    if conf:
        app_conf.update(conf)

    from monailabel.config import settings

    settings.MONAI_LABEL_APP_DIR = app_dir
    settings.MONAI_LABEL_STUDIES = studies
    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_APP_CONF = app_conf
    settings.MONAI_LABEL_SESSION_PATH = os.path.join(data_dir, "sessions")

    for k, v in settings.dict().items():
        v = json.dumps(v) if isinstance(v, list) or isinstance(v, dict) else str(v)
        os.environ[k] = v
        logging.debug(f"{k} => {v}")

    logs_dir = os.path.join(app_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    open(os.path.join(logs_dir, "app.log"), "a").close()

    from monailabel.app import app
    from monailabel.interfaces.utils.app import clear_cache

    clear_cache()
    return TestClient(app)


class BasicEndpointTestSuite(unittest.TestCase):
    client = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "radiology")
    studies = os.path.join(data_dir, "dataset", "local", "spleen")
    rand_id = random.randint(0, 9999)

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.append(cls.app_dir)
        cls.client = create_client(cls.app_dir, cls.studies, cls.data_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(cls.app_dir)


class DICOMWebEndpointTestSuite(unittest.TestCase):
    client = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data", "dataset", "dicomweb")

    settings.MONAI_LABEL_DICOMWEB_CACHE_PATH = data_dir

    app_dir = os.path.join(base_dir, "sample-apps", "radiology")
    studies = "http://faketesturl:8042/dicom-web"

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.append(cls.app_dir)
        cls.client = create_client(cls.app_dir, cls.studies, cls.data_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(cls.app_dir)


class BasicEndpointV2TestSuite(unittest.TestCase):
    client = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "radiology")
    studies = os.path.join(data_dir, "dataset", "local", "spleen")
    rand_id = random.randint(0, 9999)

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.append(cls.app_dir)
        cls.client = create_client(
            cls.app_dir, cls.studies, cls.data_dir, {"models": "deepgrow_2d,deepgrow_3d,segmentation_spleen"}
        )

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(cls.app_dir)


class BasicEndpointV3TestSuite(unittest.TestCase):
    client = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "radiology")
    studies = os.path.join(data_dir, "dataset", "local", "spleen")
    rand_id = random.randint(0, 9999)

    @classmethod
    def setUpClass(cls) -> None:
        conf = {
            "epistemic_enabled": "true",
            "epistemic_samples": "2",
            "skip_strategies": "false",
            "skip_scoring": "false",
            "models": "segmentation_spleen",
        }
        sys.path.append(cls.app_dir)
        cls.client = create_client(cls.app_dir, cls.studies, cls.data_dir, conf=conf)

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(cls.app_dir)
