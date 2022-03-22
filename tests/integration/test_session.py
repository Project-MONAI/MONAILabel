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
import unittest

import requests

from tests.integration import SERVER_URI


class EndPointSession(unittest.TestCase):
    def test_session(self):
        studies = "tests/data/dataset/local/spleen"
        files = [("files", ("spleen_3.nii.gz", open(os.path.join(studies, "spleen_3.nii.gz"), "rb")))]

        response = requests.put(f"{SERVER_URI}/session/?expiry=600", files=files)
        assert response.status_code == 200

        r = response.json()
        session_id = r["session_id"]
        session_info = r["session_info"]

        assert session_id
        assert session_info

        response = requests.get(f"{SERVER_URI}/session/{session_id}")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
