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
import hashlib
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from dicomweb_client import DICOMwebClient
from dicomweb_client.api import load_json_dataset
from expiringdict import ExpiringDict

from monailabel.datastore.local import LocalDatastore
from monailabel.datastore.utils.convert import binary_to_image, dicom_to_nifti, nifti_to_dicom_seg
from monailabel.datastore.utils.dicom import dicom_web_download_series, dicom_web_upload_dcm
from monailabel.interfaces.datastore import DefaultLabelTag

logger = logging.getLogger(__name__)


class DICOMwebClientX(DICOMwebClient):
    def _decode_multipart_message(self, response: requests.Response, stream: bool) -> Iterator[bytes]:
        content_type = response.headers["content-type"]
        media_type, *ct_info = [ct.strip() for ct in content_type.split(";")]
        if media_type.lower() != "multipart/related":
            response.headers["content-type"] = "multipart/related"
        return super()._decode_multipart_message(response, stream)


class DICOMWebDatastore(LocalDatastore):
    def __init__(self, client: DICOMwebClient, cache_path: Optional[str] = None, fetch_by_frame=False):
        self._client = client
        self._modality = "CT"
        self._fetch_by_frame = fetch_by_frame

        uri_hash = hashlib.md5(self._client.base_url.encode("utf-8")).hexdigest()
        datastore_path = (
            os.path.join(cache_path, uri_hash)
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", uri_hash)
        )
        logger.info(f"DICOMWeb Datastore (cache) Path: {datastore_path}; FetchByFrame: {fetch_by_frame}")

        self._stats_cache = ExpiringDict(max_len=100, max_age_seconds=30)
        super().__init__(datastore_path=datastore_path, auto_reload=True)

    def name(self) -> str:
        base_url: str = self._client.base_url
        return base_url

    def _to_id(self, file: str) -> Tuple[str, str]:
        extensions = [".nii", ".nii.gz", ".nrrd"]
        for extension in extensions:
            if file.endswith(extension):
                return file.replace(extension, ""), extension
        return super()._to_id(file)

    def get_image_uri(self, image_id: str) -> str:
        logger.info(f"Image ID: {image_id}")
        image_dir = os.path.realpath(os.path.join(self._datastore.image_path(), image_id))
        logger.info(f"Image Dir (cache): {image_dir}")

        if not os.path.exists(image_dir) or not os.listdir(image_dir):
            dicom_web_download_series(None, image_id, image_dir, self._client, self._fetch_by_frame)

        image_nii_gz = os.path.realpath(os.path.join(self._datastore.image_path(), f"{image_id}.nii.gz"))
        if not os.path.exists(image_nii_gz):
            image_nii_gz = dicom_to_nifti(image_dir)
            super().add_image(image_id, image_nii_gz, self._dicom_info(image_id))

        return image_nii_gz

    def get_label_uri(self, label_id: str, label_tag: str, image_id: str = "") -> str:
        if label_tag != DefaultLabelTag.FINAL:
            return super().get_label_uri(label_id, label_tag)

        logger.info(f"Label ID: {label_id} => {label_tag}")
        label_dir = os.path.realpath(os.path.join(self._datastore.label_path(label_tag), label_id))
        logger.info(f"Label Dir (cache): {label_dir}")

        if not os.path.exists(label_dir) or not os.listdir(label_dir):
            dicom_web_download_series(None, label_id, label_dir, self._client, self._fetch_by_frame)

        label_nii_gz = os.path.realpath(
            os.path.join(self._datastore.label_path(DefaultLabelTag.FINAL), f"{image_id}.nii.gz")
        )
        if not os.path.exists(label_nii_gz):
            label_nii_gz = dicom_to_nifti(label_dir, is_seg=True)
            if label_nii_gz:
                super().save_label(image_id, label_nii_gz, label_tag, self._dicom_info(label_id))

        return label_nii_gz

    def _dicom_info(self, series_id):
        meta = load_json_dataset(self._client.search_for_series(search_filters={"SeriesInstanceUID": series_id})[0])
        fields = ["StudyDate", "StudyTime", "Modality", "RetrieveURL", "PatientID", "StudyInstanceUID"]

        info = {"SeriesInstanceUID": series_id}
        for f in fields:
            info[f] = str(meta[f].value) if meta.get(f) else "UNK"
        return info

    def list_images(self) -> List[str]:
        datasets = self._client.search_for_series(search_filters={"Modality": self._modality})
        series = [str(load_json_dataset(ds)["SeriesInstanceUID"].value) for ds in datasets]
        logger.debug("Total Series: {}\n{}".format(len(series), "\n".join(series)))
        return series

    def get_labeled_images(self) -> List[str]:
        datasets = self._client.search_for_series(search_filters={"Modality": "SEG"})
        all_segs = [load_json_dataset(ds) for ds in datasets]

        image_series = []
        for seg in all_segs:
            meta = self._client.retrieve_series_metadata(
                str(seg["StudyInstanceUID"].value), str(seg["SeriesInstanceUID"].value)
            )
            seg_meta = load_json_dataset(meta[0])
            if seg_meta.get("ReferencedSeriesSequence"):
                image_series.append(str(seg_meta["ReferencedSeriesSequence"].value[0]["SeriesInstanceUID"].value))
            else:
                logger.warning(
                    f"Label Ignored:: ReferencedSeriesSequence is NOT found: {str(seg['SeriesInstanceUID'].value)}"
                )
        return image_series

    def get_unlabeled_images(self) -> List[str]:
        series = self.list_images()

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

        output_file = ""
        if label_ext == ".bin":
            output_file = binary_to_image(image_uri, label_filename)
            label_filename = output_file

        logger.info(f"Label File: {label_filename}")

        # Support DICOM-SEG uploading only final version
        if label_tag == DefaultLabelTag.FINAL:
            image_dir = os.path.realpath(os.path.join(self._datastore.image_path(), image_id))
            label_file = nifti_to_dicom_seg(image_dir, label_filename, label_info.get("label_info"))

            label_series_id = dicom_web_upload_dcm(label_file, self._client)
            label_info.update(self._dicom_info(label_series_id))
            os.unlink(label_file)

        label_id = super().save_label(image_id, label_filename, label_tag, label_info)
        logger.info("Save completed!")

        if output_file:
            os.unlink(output_file)
        return label_id

    def _download_labeled_data(self):
        datasets = self._client.search_for_series(search_filters={"Modality": "SEG"})
        all_segs = [load_json_dataset(ds) for ds in datasets]

        image_labels = []
        for seg in all_segs:
            meta = self._client.retrieve_series_metadata(
                str(seg["StudyInstanceUID"].value), str(seg["SeriesInstanceUID"].value)
            )
            seg_meta = load_json_dataset(meta[0])
            if not seg_meta.get("ReferencedSeriesSequence"):
                logger.warning(
                    f"Label Ignored:: ReferencedSeriesSequence is NOT found: {str(seg['SeriesInstanceUID'].value)}"
                )
                continue

            image_labels.append(
                {
                    "image": str(seg_meta["ReferencedSeriesSequence"].value[0]["SeriesInstanceUID"].value),
                    "label": str(seg["SeriesInstanceUID"].value),
                }
            )

        invalid = set(super().get_labeled_images()) - {image_label["image"] for image_label in image_labels}
        logger.info(f"Invalid Labels: {invalid}")
        for e in invalid:
            logger.info(f"Label {e} not exist on remote;  Remove from local")
            label_uri = super().get_label_uri(e, DefaultLabelTag.FINAL)
            if label_uri and os.path.exists(label_uri):
                shutil.rmtree(os.path.join(os.path.dirname(label_uri), e), ignore_errors=True)
                os.unlink(label_uri)

        for image_label in image_labels:
            self.get_image_uri(image_id=image_label["image"])
            self.get_label_uri(
                label_id=image_label["label"], label_tag=DefaultLabelTag.FINAL, image_id=image_label["image"]
            )

    def datalist(self, full_path=True) -> List[Dict[str, Any]]:
        self._download_labeled_data()
        return super().datalist(full_path)

    def status(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = self._stats_cache.get("stats")
        if not stats:
            stats = super().status()
            self._stats_cache["stats"] = stats
        return stats
