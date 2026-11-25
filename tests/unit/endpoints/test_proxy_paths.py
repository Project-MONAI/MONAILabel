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

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from monailabel.endpoints.proxy import proxy_dicom
from monailabel.endpoints.user.auth import DEFAULT_USER

from .context import BasicEndpointTestSuite


class RawData:
    def read(self):
        return b"xyz"


def mocked_requests(*args, **kwargs):
    return SimpleNamespace(content=b"xyz", raw=RawData(), status_code=200, headers={})


class TestProxyPaths(BasicEndpointTestSuite):
    """The proxy sends requests with the server's credentials, so a client must not be able to leave its path."""

    def _urls(self, mock):
        return [call.args[0] for call in mock.call_args_list]

    def test_regular_paths_are_forwarded(self):
        for op in ("", "wado/", "qido/", "stow/"):
            with patch("requests.get", side_effect=mocked_requests) as mock_get:
                res = self.client.get(f"/proxy/dicom/{op}studies/1.2.3/series")
                self.assertEqual(res.status_code, 200, op)
                self.assertEqual(len(self._urls(mock_get)), 1)
                self.assertIn("studies/1.2.3/series", self._urls(mock_get)[0])

    def test_traversal_is_rejected_before_anything_is_sent(self):
        # percent-encoded: not collapsed by HTTP clients/servers, so it reaches the handler as "..", "../" ...
        paths = (
            "%2e%2e/%2e%2e/datasets/other/dicomStores/x/dicomWeb/studies",
            "studies/%2E%2E/x",
            "%252e%252e/x",
            "studies/..%2f..%2fx",
            "studies/%2e%2e%5c%2e%2e%5cx",
        )
        for op in ("", "wado/", "qido/", "stow/"):
            for path in paths:
                with patch("requests.get", side_effect=mocked_requests) as mock_get, patch(
                    "requests.post", side_effect=mocked_requests
                ) as mock_post:
                    for method in (self.client.get, self.client.post):
                        res = method(f"/proxy/dicom/{op}{path}")
                        self.assertEqual(res.status_code, 400, f"{op}{path}")
                    self.assertEqual(mock_get.call_count + mock_post.call_count, 0, f"{op}{path}")

    def test_plain_traversal_is_rejected_by_the_handler(self):
        # a raw client can send "../" as is (the test client would collapse it before sending)
        for path in ("../../datasets/other/dicomStores/x/dicomWeb/studies", "studies/../../x", "studies/.."):
            for op in ("", "wado", "qido", "stow"):
                with patch("requests.get", side_effect=mocked_requests) as mock_get:
                    with self.assertRaises(HTTPException) as cm:
                        asyncio.run(proxy_dicom(SimpleNamespace(method="GET"), op, path, DEFAULT_USER))
                    self.assertEqual(cm.exception.status_code, 400)
                    self.assertEqual(mock_get.call_count, 0)


if __name__ == "__main__":
    unittest.main()
