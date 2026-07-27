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

import importlib.util
import pathlib
import tempfile
import unittest
from unittest.mock import patch


CLIENT_PATH = pathlib.Path(__file__).resolve().parents[3] / "sample-apps" / "reviewer" / "client.py"
SPEC = importlib.util.spec_from_file_location("reviewer_client", CLIENT_PATH)
reviewer_client = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reviewer_client)
LightweightReviewClient = reviewer_client.LightweightReviewClient


class MockResponse:
    status_code = 200


class TestReviewerClientTimeouts(unittest.TestCase):
    def setUp(self):
        with patch.object(LightweightReviewClient, "ping", return_value=True):
            self.client = LightweightReviewClient("http://localhost:8000", timeout=17)

    def test_update_labelinfo_uses_configured_timeout(self):
        with patch.object(reviewer_client.requests, "put", return_value=MockResponse()) as mock_put:
            self.assertTrue(self.client.update_labelinfo("case-001", "approved"))

        self.assertEqual(mock_put.call_args.kwargs.get("timeout"), 17)

    def test_save_label_uses_configured_timeout(self):
        with tempfile.NamedTemporaryFile(suffix=".nrrd") as label_file:
            with patch.object(reviewer_client.requests, "put", return_value=MockResponse()) as mock_put:
                self.assertTrue(self.client.save_label("case-001", pathlib.Path(label_file.name)))

        self.assertEqual(mock_put.call_args.kwargs.get("timeout"), 17)

    def test_create_client_without_arguments_uses_default_server_url(self):
        with patch.object(LightweightReviewClient, "ping", return_value=True):
            client = reviewer_client.create_client()

        self.assertEqual(client.server_url, "http://localhost:8000")

    def test_create_client_preserves_explicit_server_url(self):
        with patch.object(LightweightReviewClient, "ping", return_value=True):
            client = reviewer_client.create_client("http://example.com:8001/")

        self.assertEqual(client.server_url, "http://example.com:8001")


if __name__ == "__main__":
    unittest.main()