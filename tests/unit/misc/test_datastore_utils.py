import hashlib
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from pydicom import Dataset
import pydicom

from monailabel.datastore.utils.dicom import dicom_web_download_series, dicom_web_upload_dcm


study_id = "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996072"
series_id = "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721"
instance_id = "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656724.dcm"


class Client(object):

    def __init__(self, cache_path) -> None:
        self.cache_path = cache_path

    def store_instances(self, *args, **kwargs):
        ds = Dataset()
        ds.add_new("0x001000200", "CS", f"/series/{series_id}")
        return ds

    def search_for_series(self, *args, **kwargs):
        return [1, ]

    def retrieve_series(self, *args, **kwargs):
        series_dir = os.path.join(self.cache_path, series_id)
        dcms = os.listdir(series_dir)
        ds = []
        for dcm in dcms:
            ds.append(pydicom.dcmread(os.path.join(series_dir, dcm)))
        return ds


# class ConvertUtils(unittest.TestCase):
#     def test_dicom_to_nifti(
#         self,
#     ):
#         pass

#     def test_binary_to_image(
#         self,
#     ):
#         pass

#     def test_itk_image_to_dicom_seg(
#         self,
#     ):
#         pass

#     def test_itk_dicom_seg_to_image(
#         self,
#     ):
#         pass


class DicomUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        cls.data_dir = os.path.join(cls.base_dir, "tests", "data", "dataset", "dicomweb")
        cls.studies = "http://faketesturl:8042/dicom-web"
        cls.cache_path = os.path.join(cls.data_dir, hashlib.md5(cls.studies.encode("utf-8")).hexdigest())

        cls.client = Client(cls.cache_path)

    @patch("monailabel.datastore.utils.dicom.load_json_dataset")
    def test_dicom_web_download_series(self, load_json_dataset):
        rand_dir = str(uuid4().hex)
        save_dir = os.path.join(self.cache_path, rand_dir)

        ds = Dataset()
        ds.add_new('0x0020000D', 'UI', study_id)
        load_json_dataset.return_value = ds

        dicom_web_download_series(None, series_id, save_dir, self.client)

        downloaded_items = os.listdir(save_dir)
        original_items = os.listdir(os.path.join(self.cache_path, series_id))
        self.assertCountEqual(downloaded_items, original_items)

    def test_dicom_web_upload_dcm(self):
        input_file = os.path.join(self.cache_path, series_id, instance_id)

        ret_series_id = dicom_web_upload_dcm(input_file, self.client)
        self.assertEqual(ret_series_id, "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721")


if __name__ == "__main__":
    unittest.main()
