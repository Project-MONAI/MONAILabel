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
import argparse
import os
import unittest

from monailabel.datastore.xnat import XNATDatastore

base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class ProjectResponse:
    def json(self):
        return {"ResultSet": {"Result": [{"ID": "project1"}]}}


class ExperimentResponse:
    def json(self):
        return {"ResultSet": {"Result": [{"ID": "experiment1"}]}}


xml_response = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<xnat:MRSession xmlns:xnat="http://nrg.wustl.edu/xnat">
<xnat:subject_ID>CENTRAL_S00358</xnat:subject_ID>
<xnat:scan ID="3" type="AX FRFSE-XL T2"/>
</xnat:MRSession>
"""


class XNATDatastorMocked(XNATDatastore):
    def _request_get(self, url):
        if "JSESSION?CSRF=true" in url:
            return argparse.Namespace(ok=True, content=b"xyzc=deffad")
        if "projects?format=json" in url:
            return ProjectResponse()
        if "experiments?format=json" in url:
            return ExperimentResponse()
        if "experiment1?format=xml" in url:
            return argparse.Namespace(content=xml_response)
        if "format=zip" in url:
            with open(os.path.join(base_dir, "downloads", "dicom.zip"), mode="rb") as file:
                content = file.read()
            return argparse.Namespace(ok=True, content=content)
        if "format=xml" in url:
            return argparse.Namespace(ok=True)
        return None

    def _request_post(self, url):
        return argparse.Namespace(ok=True)


class TestXNAT(unittest.TestCase):
    def test_fetch(self):
        xnat = XNATDatastorMocked("http://xnat.com", cache_path=os.path.join(base_dir, "data", "xnat_cache"))
        xnat.get_image_info("abcd/xyz/1234/def")
        xnat.get_image("abcd/xyz/1234/def")
        xnat.list_images()
