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

import os
import tempfile
import unittest
from unittest.mock import patch

from dicomweb_client import DICOMwebClient


class Instance(dict):
    def save_as(self, f):
        pass

    def iterall(self):
        return [SOPInstanceUID("/series/xyz")]


class SOPInstanceUID:
    def __init__(self, value="xyz"):
        self.value = value

    def val(self):
        return self.value


class MockDICOMwebClient(DICOMwebClient):
    def __init__(self):
        pass

    def retrieve_series(self, *args, **kwargs):
        instance = Instance()
        instance["SOPInstanceUID"] = SOPInstanceUID()
        return [instance]

    def store_instances(self, *args, **kwargs):
        return Instance()


class TestDicom(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    local_dataset = os.path.join(base_dir, "data", "dataset", "local", "spleen")
    dicom_dataset = os.path.join(base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd")

    def xtest_generate_key(self):
        from monailabel.datastore.utils.dicom import generate_key

        generate_key("xyz", "1.2.", "3.2")

    @patch("monailabel.utils.others.generic.run_command")
    def test_get_scu(self, f1):
        f1.return_value = 0
        from monailabel.datastore.utils.dicom import get_scu

        get_scu("xyz", "abc")

    @patch("monailabel.utils.others.generic.run_command")
    def test_store_scu(self, f2):
        f2.return_value = 0
        from monailabel.datastore.utils.dicom import store_scu

        store_scu(self.local_dataset)

    def test_dicom_web_download_series(self):
        from monailabel.datastore.utils.dicom import dicom_web_download_series

        with tempfile.TemporaryDirectory() as d:
            dicom_web_download_series("xyz", "abc", d, MockDICOMwebClient())

    @patch("monailabel.datastore.utils.dicom.dcmread")
    def test_dicom_web_upload_dcm(self, f3):
        f3.return_value = "xyz"

        from monailabel.datastore.utils.dicom import dicom_web_upload_dcm

        dicom_web_upload_dcm("xyz", MockDICOMwebClient())


if __name__ == "__main__":
    unittest.main()
