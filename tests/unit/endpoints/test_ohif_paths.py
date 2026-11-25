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

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from monailabel.endpoints import ohif


class TestOhifFileServing(unittest.TestCase):
    """/ohif serves files from the OHIF directory only; everything else gets the viewer's index page."""

    def setUp(self):
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="monailabel_ohif_"))
        self.ohif_dir = os.path.join(self.base, "static", "ohif")
        os.makedirs(os.path.join(self.ohif_dir, "assets"))
        for name in ("index.html", "app-config.js", os.path.join("assets", "logo.svg")):
            with open(os.path.join(self.ohif_dir, name), "w") as fp:
                fp.write(name)
        # files next to / above the OHIF directory that must never be served
        for name in (os.path.join("static", "secret.txt"), "secret.txt", os.path.join("static", "ohif-other.txt")):
            with open(os.path.join(self.base, name), "w") as fp:
                fp.write("secret")

        patcher = patch.object(ohif, "OHIF_DIR", self.ohif_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _served(self, path):
        return os.path.relpath(ohif.get_ohif(path).path, self.ohif_dir)

    def test_serves_existing_files_including_nested_ones(self):
        self.assertEqual(self._served("app-config.js"), "app-config.js")
        self.assertEqual(self._served("assets/logo.svg"), os.path.join("assets", "logo.svg"))
        self.assertEqual(self._served("assets/../app-config.js"), "app-config.js")

    def test_unknown_paths_and_directories_get_the_index_page(self):
        for path in ("", "viewer/1.2.3", "missing.js", "assets", "assets/"):
            with self.subTest(path=path):
                self.assertEqual(self._served(path), "index.html")

    def test_paths_outside_the_ohif_directory_get_the_index_page(self):
        outside = [
            "../secret.txt",
            "../../secret.txt",
            "assets/../../secret.txt",
            "../ohif-other.txt",
            "../../static/secret.txt",
            os.path.join(self.base, "secret.txt"),
            "/etc/passwd",
            "..\\..\\secret.txt",
            "..%2f..%2fsecret.txt",
            "app-config.js\x00../../secret.txt",
        ]
        for path in outside:
            with self.subTest(path=path):
                self.assertEqual(self._served(path), "index.html")

    def test_symlink_pointing_outside_is_not_followed(self):
        link = os.path.join(self.ohif_dir, "link.txt")
        try:
            os.symlink(os.path.join(self.base, "secret.txt"), link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not supported")
        self.assertEqual(self._served("link.txt"), "index.html")

    def test_missing_index_page_is_a_404(self):
        os.unlink(os.path.join(self.ohif_dir, "index.html"))
        with self.assertRaises(HTTPException) as e:
            ohif.get_ohif("../secret.txt")
        self.assertEqual(e.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
