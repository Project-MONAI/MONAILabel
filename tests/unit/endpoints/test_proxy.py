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
from unittest.mock import MagicMock, patch

from .context import BasicEndpointTestSuite


class MockHttpClient(MagicMock):
    def __init__(self, auth):
        pass

    async def get(self, url, **kwargs):
        return SimpleNamespace(content=b"xyz", status_code=400)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def mock_http_client(auth):
    return MockHttpClient(auth)


@patch("httpx.AsyncClient", new=mock_http_client)
class TestEndPointLogs(BasicEndpointTestSuite):
    def test_proxy(self):
        self.client.get("/proxy/dicom/studies")


if __name__ == "__main__":
    unittest.main()
