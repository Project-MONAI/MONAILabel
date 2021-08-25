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
import base64
import hashlib
import logging
import os
import pathlib
from typing import Any, Dict, List

import SimpleITK
from dicomweb_client import DICOMwebClient
from dicomweb_client.api import load_json_dataset

from monailabel.utils.datastore.dicom.util import binary_to_image, get_scu, nifti_to_dicom_seg, score_scu
from monailabel.utils.datastore.local_v2 import LocalDatastore
from monailabel.utils.others.generic import file_checksum

logger = logging.getLogger(__name__)


class DICOMWebCache(LocalDatastore):
    def __init__(self, client: DICOMwebClient):
        self._client = client
        uri_hash = hashlib.md5(self._client.base_url.encode("utf-8")).hexdigest()
        datastore_path = os.path.join(pathlib.Path.home(), ".cache", "monailabel", uri_hash)

        self.use_ae = True

        super().__init__(datastore_path=datastore_path, auto_reload=True)

    def name(self) -> str:
        return self._client.base_url

    def get_image_uri(self, image_id: str) -> str:
        logger.info(f"Image ID: {image_id}")
        image_dir = os.path.realpath(os.path.join(self._datastore.image_path(), image_id))
        logger.info(f"Image Dir (cache): {image_dir}")

        if not os.path.exists(image_dir) or not os.listdir(image_dir):
            os.makedirs(image_dir, exist_ok=True)
            if self.use_ae:
                get_scu(image_id, image_dir, query_level="SERIES")
            else:
                # Limitation for DICOMWeb Client as it needs StudyInstanceUID to fetch series
                meta = load_json_dataset(
                    self._client.search_for_series(search_filters={"SeriesInstanceUID": image_id})[0]
                )
                instances = self._client.retrieve_series(str(meta["StudyInstanceUID"].value), image_id)
                for instance in instances:
                    instance_id = instance.SOPInstanceUID
                    file_name = os.path.join(image_dir, f"{instance_id}.dcm")
                    instance.save_as(file_name)

        # TODO:: BUG In MONAI? Currently can not load DICOM through ITK Loader
        image_id_encoded = base64.standard_b64encode(image_id.encode("utf-8")).decode("utf-8")
        image_nii_gz = os.path.realpath(os.path.join(self._datastore.image_path(), image_id_encoded)) + ".nii.gz"
        if not os.path.exists(image_nii_gz):
            reader = SimpleITK.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(image_dir)
            reader.SetFileNames(dicom_names)

            image = reader.Execute()
            logger.info(f"Image size: {image.GetSize()}")

            SimpleITK.WriteImage(image, image_nii_gz)
        return image_nii_gz

        # series_dir = tempfile.TemporaryDirectory()
        # get_scu(image_id, series_dir, query_level="SERIES")

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        """
        Get the image information for the given image id

        :param image_id: the desired image id
        :return: image info as a list of dictionaries Dict[str, Any]
        """
        path = self.get_image_uri(image_id)
        return {
            "checksum": file_checksum(path),
            "name": os.path.basename(path),
            "path": path,
        }

    def get_labeled_images(self) -> List[str]:
        datasets = self._client.search_for_series(search_filters={"Modality": "SEG"})
        all_segs = [load_json_dataset(ds) for ds in datasets]

        image_series = []
        for seg in all_segs:
            meta = self._client.retrieve_series_metadata(
                str(seg["StudyInstanceUID"].value), str(seg["SeriesInstanceUID"].value)
            )
            seg_meta = load_json_dataset(meta[0])
            image_series.append(str(seg_meta["ReferencedSeriesSequence"].value[0]["SeriesInstanceUID"].value))
        return image_series

    def get_unlabeled_images(self) -> List[str]:
        datasets = self._client.search_for_series(search_filters={"Modality": "CT"})
        series = [str(load_json_dataset(ds)["SeriesInstanceUID"].value) for ds in datasets]
        logger.info("Total Series: {}\n{}".format(len(series), "\n".join(series)))

        seg_series = self.get_labeled_images()
        logger.info("Total Series (with seg): {}\n{}".format(len(seg_series), "\n".join(seg_series)))
        return list(set(series) - set(seg_series))

    def save_label(
        self, image_id: str, label_filename: str, label_tag: str, label_info: Dict[str, Any], label_id: str = ""
    ) -> str:
        logger.info(f"Input - Image Id: {image_id}")
        logger.info(f"Input - Label File: {label_filename}")
        logger.info(f"Input - Label Tag: {label_tag}")
        logger.info(f"Input - Label Info: {label_info}")

        image_uri = self.get_image_uri(image_id)
        label_ext = "".join(pathlib.Path(label_filename).suffixes)

        output_file = None
        if label_ext == ".bin":
            output_file = binary_to_image(image_uri, label_filename)
            label_filename = output_file

        logger.info(f"Label File: {label_filename}")
        # res = super().save_label(image_id, label_filename, label_tag, label_info, label_id)

        image_dir = os.path.realpath(os.path.join(self._datastore.image_path(), image_id))
        label_file = nifti_to_dicom_seg(image_dir, label_filename, label_info)
        score_scu(label_file)  # TODO:: Get new Series ID for this label which is uploaded

        image_id_encoded = base64.standard_b64encode(image_id.encode("utf-8")).decode("utf-8")
        super().save_label(image_id_encoded, label_filename, label_tag, label_info)

        os.unlink(label_file)
        if output_file:
            os.unlink(output_file)
        return ""
