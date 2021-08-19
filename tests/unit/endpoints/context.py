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

import os
import random
import sys
import unittest

from fastapi.testclient import TestClient

from monailabel.config import settings
from monailabel.main import app


class BasicEndpointTestSuite(unittest.TestCase):
    client = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "deepedit_left_atrium")
    studies = os.path.join(data_dir, "dataset", "heart")
    rand_id = random.randint(0, 9999)

    @classmethod
    def setUpClass(cls) -> None:
        settings.APP_DIR = cls.app_dir
        settings.STUDIES = cls.studies
        settings.DATASTORE_AUTO_RELOAD = False

        sys.path.append(settings.APP_DIR)
        sys.path.append(os.path.join(settings.APP_DIR, "lib"))

        logs_dir = os.path.join(cls.app_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        open(os.path.join(logs_dir, "app.log"), "a").close()

        cls.client = TestClient(app)
