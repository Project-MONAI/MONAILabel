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
import os
import pathlib
import unittest
from unittest.mock import patch

APP_PATH = pathlib.Path(__file__).resolve().parents[3] / "sample-apps" / "reviewer" / "app.py"
SPEC = importlib.util.spec_from_file_location("reviewer_app", APP_PATH)
reviewer_app = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reviewer_app)
ReviewerApp = reviewer_app.ReviewerApp


class TestReviewerAppConfig(unittest.TestCase):
    def test_conf_mode_is_preserved_without_env_override(self):
        app = ReviewerApp.__new__(ReviewerApp)
        with patch.dict(os.environ, {}, clear=False):
            config = app._load_review_config("/tmp/studies", {"mode": "standalone"})

        self.assertEqual(config["mode"], "standalone")

    def test_env_mode_overrides_configured_mode(self):
        app = ReviewerApp.__new__(ReviewerApp)
        with patch.dict(os.environ, {"MONAI_LABEL_REVIEW_MODE": "review"}, clear=False):
            config = app._load_review_config("/tmp/studies", {"mode": "standalone"})

        self.assertEqual(config["mode"], "review")


if __name__ == "__main__":
    unittest.main()
