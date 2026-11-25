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
from unittest.mock import MagicMock, patch

from monailabel.datastore.dicom import DICOMWebDatastore
from monailabel.interfaces.datastore import DefaultLabelTag


class TestDICOMWebDatastorePathConfinement(unittest.TestCase):
    """Series ids must name a directory inside the DICOMweb cache before anything is read or sent to the PACS."""

    SERIES = "1.2.826.0.1.3680043.8.498.1"

    def setUp(self):
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="monailabel_dicomweb_confine_"))
        self.client = MagicMock()
        self.client.base_url = "http://pacs.invalid/dicom-web"
        self.datastore = DICOMWebDatastore(
            self.client, search_filter={}, cache_path=os.path.join(self.base, "cache"), convert_to_nifti=False
        )
        self.image_root = os.path.realpath(self.datastore._datastore.image_path())

        # a cached series, and a directory outside the cache that a crafted id would point to
        self.series_dir = os.path.join(self.image_root, self.SERIES)
        os.makedirs(self.series_dir)
        with open(os.path.join(self.series_dir, "1.dcm"), "wb") as fp:
            fp.write(b"dicom")

        self.outside = os.path.join(self.base, "outside")
        os.makedirs(self.outside)
        with open(os.path.join(self.outside, "other.dcm"), "wb") as fp:
            fp.write(b"other")

        self.label_file = os.path.join(self.base, "label.nii.gz")
        with open(self.label_file, "wb") as fp:
            fp.write(b"label")

    def tearDown(self):
        observer = getattr(self.datastore, "_observer", None)
        if observer is not None:
            observer.stop()
        shutil.rmtree(self.base, ignore_errors=True)

    def _bad_ids(self):
        return [
            os.path.relpath(self.outside, self.image_root),
            self.outside,
            "..",
            ".",
            "",
            f"{self.SERIES}/..",
            f"{self.SERIES}/../../outside",
            "labels/final",
        ]

    def test_get_image_uri_rejects_ids_outside_the_series_dir(self):
        for image_id in self._bad_ids():
            with self.subTest(image_id=image_id):
                with self.assertRaises(ValueError):
                    self.datastore.get_image_uri(image_id)
        self.client.search_for_series.assert_not_called()
        self.client.retrieve_series.assert_not_called()

    def test_get_label_uri_rejects_ids_outside_the_series_dir(self):
        label_root = self.datastore._datastore.label_path(DefaultLabelTag.FINAL)
        for label_id in self._bad_ids() + [os.path.relpath(self.outside, label_root)]:
            with self.subTest(label_id=label_id):
                with self.assertRaises(ValueError):
                    self.datastore.get_label_uri(label_id, DefaultLabelTag.FINAL)
        self.client.search_for_series.assert_not_called()
        self.client.retrieve_series.assert_not_called()

    def test_save_label_rejects_ids_outside_the_series_dir_before_anything_is_sent(self):
        with (
            patch("monailabel.datastore.dicom.nifti_to_dicom_seg") as to_seg,
            patch("monailabel.datastore.dicom.dicom_web_upload_dcm") as upload,
        ):
            for image_id in self._bad_ids():
                with self.subTest(image_id=image_id):
                    with self.assertRaises(ValueError):
                        self.datastore.save_label(image_id, self.label_file, DefaultLabelTag.FINAL, {})
            to_seg.assert_not_called()
            upload.assert_not_called()
        self.client.store_instances.assert_not_called()
        self.assertTrue(os.path.exists(os.path.join(self.outside, "other.dcm")))

    def test_cached_series_is_still_served(self):
        self.assertEqual(self.datastore.get_image_uri(self.SERIES), self.series_dir)
        self.client.retrieve_series.assert_not_called()

    def test_save_label_final_uses_the_cached_series(self):
        seg_file = os.path.join(self.base, "seg.dcm")
        with open(seg_file, "wb") as fp:
            fp.write(b"seg")

        with (
            patch("monailabel.datastore.dicom.nifti_to_dicom_seg", return_value=seg_file) as to_seg,
            patch("monailabel.datastore.dicom.dicom_web_upload_dcm", return_value="1.2.3") as upload,
            patch.object(DICOMWebDatastore, "get_image_info", return_value={}),
            patch("monailabel.datastore.local.LocalDatastore.save_label", return_value=self.SERIES),
        ):
            self.assertEqual(
                self.datastore.save_label(self.SERIES, self.label_file, DefaultLabelTag.FINAL, {}), self.SERIES
            )
        self.assertEqual(to_seg.call_args.args[0], self.series_dir)
        upload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
