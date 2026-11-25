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

import hashlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from monailabel.endpoints import datastore
from monailabel.utils.others.generic import file_checksum


class FakeDatastore:
    def __init__(self, image):
        self.image = image

    def get_image_uri(self, image_id):
        return self.image


class FakeApp:
    def __init__(self, image):
        self._datastore = FakeDatastore(image)

    def datastore(self):
        return self._datastore


class TestImageChecksum(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="monailabel_checksum_")
        self.image = os.path.join(self.base, "image.nii.gz")
        with open(self.image, "wb") as fp:
            fp.write(b"image-bytes")
        self.hex = hashlib.sha256(b"image-bytes").hexdigest()

        patcher = patch.object(datastore, "app_instance", return_value=FakeApp(self.image))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_parse_checksum(self):
        self.assertEqual(datastore.parse_checksum("sha256:ABC"), ("SHA256", "abc"))
        self.assertEqual(datastore.parse_checksum("MD5:abc"), ("MD5", "abc"))
        self.assertEqual(datastore.parse_checksum("abc"), ("SHA256", "abc"))

    def test_checksum_as_written_by_file_checksum_matches(self):
        for check_sum in (file_checksum(self.image), self.hex, f"sha256:{self.hex.upper()}", f"SHA256:{self.hex}"):
            with self.subTest(check_sum=check_sum):
                self.assertEqual(datastore.download_image("image", check_only=True, check_sum=check_sum), {})

    def test_checksum_of_other_content_does_not_match(self):
        other = hashlib.sha256(b"other").hexdigest()
        for check_sum in (other, f"SHA256:{other}", "SHA256:", "MD5:" + hashlib.md5(b"other").hexdigest()):
            with self.subTest(check_sum=check_sum):
                with self.assertRaises(HTTPException) as e:
                    datastore.download_image("image", check_only=True, check_sum=check_sum)
                self.assertEqual(e.exception.status_code, 404)

    def test_other_algorithms(self):
        md5 = hashlib.md5(b"image-bytes").hexdigest()
        self.assertEqual(datastore.download_image("image", check_only=True, check_sum=f"MD5:{md5}"), {})

    def test_unsupported_algorithm_is_rejected(self):
        with self.assertRaises(HTTPException) as e:
            datastore.download_image("image", check_only=True, check_sum="CRC32:abcd")
        self.assertEqual(e.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
