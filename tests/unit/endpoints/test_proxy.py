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
from types import SimpleNamespace
from unittest.mock import patch

from .context import BasicEndpointTestSuite


class RawData:
    def read(self):
        return b"xyz"


def mocked_requests_get(*args, **kwargs):
    return SimpleNamespace(content=b"xyz", raw=RawData(), status_code=400, headers={})


@patch("requests.get", side_effect=mocked_requests_get)
class TestEndPointLogs(BasicEndpointTestSuite):
    def test_proxy(self, mock_get):
        self.client.get("/proxy/dicom/studies")


if __name__ == "__main__":
    unittest.main()
