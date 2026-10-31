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

    def test_proxy_forwards_query(self, mock_get):
        query = "PatientName=*Horse*&limit=101&offset=0&includefield=00081030%2C00080060"
        self.client.get(f"/proxy/dicom/qido/studies?{query}")
        self.assertTrue(mock_get.call_args.args[0].endswith(f"/studies?{query}"), mock_get.call_args.args[0])

        self.client.get("/proxy/dicom/qido/studies")
        self.assertTrue(mock_get.call_args.args[0].endswith("/studies"), mock_get.call_args.args[0])

    @patch("requests.post", side_effect=mocked_requests_get)
    def test_proxy_post_forwards_query(self, mock_post, mock_get):
        query = "PatientName=*Horse*&limit=101"
        self.client.post(f"/proxy/dicom/qido/studies?{query}")
        self.assertTrue(mock_post.call_args.args[0].endswith(f"/studies?{query}"), mock_post.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
