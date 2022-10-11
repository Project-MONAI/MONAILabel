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
import unittest
from typing import Dict
from unittest.mock import patch

import pydicom

from monailabel.datastore.dicom import DICOMWebDatastore

from .context import DICOMWebEndpointTestSuite


def search_for_series(data_dir, **kwargs):
    search_filters = None
    if "search_filters" in kwargs:
        search_filters = kwargs["search_filters"]

    modality = "CT"
    if search_filters and "Modality" in search_filters:
        modality = search_filters["Modality"]

    response_dir = os.path.join(data_dir, "responses")
    if modality == "CT":
        with open(os.path.join(response_dir, "search_for_series_ct.json")) as f:
            resp = json.load(f)
    elif modality == "SEG":
        with open(os.path.join(response_dir, "search_for_series_seg.json")) as f:
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

    @patch("monailabel.interfaces.app.DICOMwebClientX")
    def test_datastore_stats(self, dwc):
        cache_path = os.path.join(self.data_dir, hashlib.md5(self.studies.encode("utf-8")).hexdigest())
        dwc.return_value.base_url = self.studies
        dwc.return_value.search_for_series = lambda **kwargs: search_for_series(self.data_dir, **kwargs)
        dwc.return_value.retrieve_series_metadata = lambda *args, **kwargs: retrieve_series_metadata(
            self.data_dir, *args, **kwargs
        )
        dwc.return_value.load_json_dataset = lambda *args, **kwargs: retrieve_series(cache_path, *args, **kwargs)

        response = self.client.get("/datastore/?output=stats")
        self.assertEqual(response.status_code, 200)

        res = response.json()
        total = res["total"]
        completed = res["completed"]
        label_tags = res["label_tags"]
        self.assertEqual(total, 8)
        self.assertEqual(completed, 3)
        self.assertEqual(label_tags["original"], 2)
        self.assertEqual(label_tags["final"], 1)

    @patch("monailabel.interfaces.app.DICOMwebClientX")
    @patch("monailabel.datastore.dicom.dicom_web_download_series")
    def test_datastore_all(self, dicom_web_download_series, dwc):
        dicom_web_download_series.return_value = lambda *args: None

        cache_path = os.path.join(self.data_dir, hashlib.md5(self.studies.encode("utf-8")).hexdigest())
        dwc.return_value.base_url = self.studies
        dwc.return_value.search_for_series = lambda **kwargs: search_for_series(self.data_dir, **kwargs)
        dwc.return_value.retrieve_series_metadata = lambda *args, **kwargs: retrieve_series_metadata(
            self.data_dir, *args, **kwargs
        )
        dwc.return_value.load_json_dataset = lambda *args, **kwargs: retrieve_series(cache_path, *args, **kwargs)

        response = self.client.get("/datastore/?output=all")
        self.assertEqual(response.status_code, 200)

        res = response.json()
        for k in res["objects"].keys():
            self.assertIn(
                k,
                [
                    "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087",
                    "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
                    "1.2.826.0.1.3680043.8.274.1.1.8323329.686521.1629744176.620266",
                ],
            )

    @patch("monailabel.interfaces.app.DICOMwebClientX")
    @patch("monailabel.datastore.dicom.dicom_web_download_series")
    def test_save_label(self, dicom_web_download_series, dwc):
        dicom_web_download_series.return_value = lambda *args: None

        cache_path = os.path.join(self.data_dir, hashlib.md5(self.studies.encode("utf-8")).hexdigest())
        dwc.return_value.base_url = self.studies
        dwc.return_value.search_for_series = lambda **kwargs: search_for_series(self.data_dir, **kwargs)
        dwc.return_value.retrieve_series_metadata = lambda *args, **kwargs: retrieve_series_metadata(
            self.data_dir, *args, **kwargs
        )
        dwc.return_value.load_json_dataset = lambda *args, **kwargs: retrieve_series(cache_path, *args, **kwargs)

        image_id = "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721"
        image_file = os.path.join(self.data_dir, f"{image_id}.nii.gz")

        test_tag = "test"
        with open(os.path.join(self.data_dir, "labels_to_upload", f"{image_id}.nii.gz"), "rb") as f:
            response = self.client.put(
                f"/datastore/label?image={image_id}&tag={test_tag}", files={"label": (image_file, f)}
            )
            self.assertEqual(response.status_code, 200)
            res = response.json()
            self.assertEqual(res["image"], image_id)
            self.assertEqual(res["label"], image_id)

        response = self.client.get("/datastore/?output=stats")
        self.assertEqual(response.status_code, 200)
        res = response.json()
        label_tags = res["label_tags"]
        self.assertEqual(label_tags[test_tag], 1)

    @patch("monailabel.interfaces.app.DICOMwebClientX")
    @patch("monailabel.datastore.dicom.dicom_web_download_series")
    def test_datastore_train(self, dicom_web_download_series, dwc):
        dicom_web_download_series.return_value = lambda *args: None

        cache_path = os.path.join(self.data_dir, hashlib.md5(self.studies.encode("utf-8")).hexdigest())
        dwc.return_value.base_url = self.studies
        dwc.return_value.search_for_series = lambda **kwargs: search_for_series(self.data_dir, **kwargs)
        dwc.return_value.retrieve_series_metadata = lambda *args, **kwargs: retrieve_series_metadata(
            self.data_dir, *args, **kwargs
        )
        dwc.return_value.load_json_dataset = lambda *args, **kwargs: retrieve_series(cache_path, *args, **kwargs)

        with patch.object(DICOMWebDatastore.__init__, "__defaults__", (None, self.data_dir, False)):
            response = self.client.get("/datastore/?output=train")
            self.assertEqual(response.status_code, 200)
            res = response.json()
            assert res


if __name__ == "__main__":
    unittest.main()
