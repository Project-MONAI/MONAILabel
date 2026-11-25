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
import unittest

from monailabel.datastore.xnat import XNATDatastore
from monailabel.tasks.train.basic_train import BasicTrainTask
from monailabel.utils.others.generic import has_path_traversal, is_googleapis_url


class TestPathTraversalCheck(unittest.TestCase):
    def test_traversal_is_detected(self):
        for path in (
            "..",
            ".",
            "../x",
            "a/../b",
            "a/..",
            "a/./b",
            "a\\..\\b",
            "%2e%2e/x",
            "%2E%2E%2Fx",
            "a/%2e%2e/b",
            "%252e%252e/x",
            "%25252e%25252e/x",
            "a%00b",
            "a/%00",
        ):
            self.assertTrue(has_path_traversal(path), path)

    def test_regular_paths_are_accepted(self):
        for path in (
            "",
            "studies",
            "studies/1.2.840.113619/series/1.2.3/instances/4.5/frames/1",
            "studies?limit=5",
            "a..b/c",
            "...",
            ".hidden/x",
            "%E2%82%AC",
        ):
            self.assertFalse(has_path_traversal(path), path)


class TestGoogleApisHost(unittest.TestCase):
    def test_host_must_be_googleapis(self):
        for url in (
            "https://healthcare.googleapis.com/v1/projects/p/locations/l/datasets/d/dicomStores/s/dicomWeb",
            "https://googleapis.com/x",
            "https://HEALTHCARE.GOOGLEAPIS.COM/x",
        ):
            self.assertTrue(is_googleapis_url(url), url)
        for url in (
            "",
            None,
            "http://localhost:8042/dicom-web",
            "https://orthanc.example.org/googleapis.com",
            "https://googleapis.com.example.org/x",
            "https://notgoogleapis.com/x",
            "https://user:googleapis.com@example.org/x",
            "http://example.org/?u=https://healthcare.googleapis.com",
        ):
            self.assertFalse(is_googleapis_url(url), url)


class TestXnatImageId(unittest.TestCase):
    def test_id_fields(self):
        ds = object.__new__(XNATDatastore)
        self.assertEqual(
            ds._id_to_fields("Test/XNAT01_S00003/XNAT01_E00004/1_2_826"),
            ("Test", "XNAT01_S00003", "XNAT01_E00004", "1_2_826"),
        )
        for image_id in (
            "../../x/y/z",
            "a/b/c/..",
            "a/../c/d",
            "a/b/c",
            "a/b/c/d/e",
            "/a/b/c",
            "a/b/c/",
            "a/b/c/d\\e",
            "",
        ):
            with self.assertRaises(ValueError, msg=image_id):
                ds._id_to_fields(image_id)


class TestTrainOutputName(unittest.TestCase):
    def test_name_stays_inside_the_model_dir(self):
        stub = type("Stub", (BasicTrainTask,), {m: lambda *a, **k: None for m in BasicTrainTask.__abstractmethods__})
        task = object.__new__(stub)
        task._model_dir = os.path.join(os.sep, "models")
        request = {"name": "train_01", "run_id": "20260101_000000"}
        self.assertEqual(task.get_output_dir(request), os.path.join(os.sep, "models", "train_01"))
        self.assertEqual(
            task.get_cache_dir(request), os.path.join(os.sep, "models", "train_01", "cache_20260101_000000")
        )
        for name in ("..", "../x", "../../etc", "/etc", "a/b", ".", "", "a\\b"):
            with self.assertRaises(ValueError, msg=name):
                task.get_output_dir({"name": name})
            with self.assertRaises(ValueError, msg=name):
                task.get_cache_dir({"name": name, "run_id": "1"})


if __name__ == "__main__":
    unittest.main()
