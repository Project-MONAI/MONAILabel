# Copyright 2021 MONAI Consortium
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
import json
import os
from typing import Dict
import unittest
from unittest.mock import patch

import pydicom
from context import DICOMWebEndpointTestSuite

import monailabel


def search_for_series(data_dir, **kwargs):
    search_filters = None
    if "search_filters" in kwargs:
        search_filters = kwargs["search_filters"]

    modality = "CT"
    if "Modality" in search_filters:
        modality = search_filters["Modality"]

    response_dir = os.path.join(data_dir, "responses")
    if modality == "CT":
        with open(os.path.join(response_dir, "search_for_series_ct.json"), "r") as f:
            resp = json.load(f)
    elif modality == "SEG":
        with open(os.path.join(response_dir, "search_for_series_seg.json"), "r") as f:
            resp = json.load(f)

    return resp


def retrieve_series(cache_path, **kwargs):
    pass


def retrieve_series_metadata(data_dir, study_id, series_id):
    meta = []
    with open(os.path.join(data_dir, "responses", "retrieve_series_meta.json")) as f:
        meta = json.load(f)
    return meta


def load_json_dataset(dataset: Dict[str, dict]):
    return pydicom.dataset.Dataset.from_json(dataset)


class EndPointDICOMWebDatastore(DICOMWebEndpointTestSuite):
    def setUp(self) -> None:
        pass

    @patch('monailabel.interfaces.app.DICOMwebClient')
    def test_001_datastore(self, dwc):

        cache_path = os.path.join(self.data_dir, hashlib.md5(self.studies.encode("utf-8")).hexdigest())
        dwc.return_value.base_url = self.studies
        dwc.return_value.search_for_series = lambda **kwargs: search_for_series(self.data_dir, **kwargs)
        # dwc.return_value.retrieve_series = lambda *args, **kwargs: retrieve_series(cache_path, *args, **kwargs)
        dwc.return_value.retrieve_series_metadata = lambda *args, **kwargs: retrieve_series_metadata(
            self.data_dir, *args, **kwargs)
        dwc.return_value.load_json_dataset = lambda *args, **kwargs: retrieve_series(cache_path, *args, **kwargs)

        with patch.object(monailabel.utils.datastore.dicom.cache.DICOMWebCache.__init__,
                          '__defaults__', (None, self.data_dir)):

            response = self.client.get("/datastore/")
            assert response.status_code == 200

            res = response.json()
            total = res["total"]
            completed = res["completed"]
            label_tags = res["label_tags"]
            assert total == 8
            assert completed == 3
            assert label_tags['original'] == 2
            assert label_tags['final'] == 1


if __name__ == "__main__":
    unittest.main()
