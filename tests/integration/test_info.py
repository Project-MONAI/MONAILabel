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

import unittest

import requests

from . import SERVER_URI


class EndPointInfo(unittest.TestCase):
    def test_info(self):
        response = requests.get(f"{SERVER_URI}/info/")
        assert response.status_code == 200

        # check if following fields exist in the response
        res = response.json()
        for f in ["version", "name", "labels"]:
            assert res[f]


if __name__ == "__main__":
    unittest.main()
