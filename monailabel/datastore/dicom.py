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
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from cachetools import TTLCache, cached
from dicomweb_client import DICOMwebClient
from pydicom.dataset import Dataset

from monailabel.config import settings
from monailabel.datastore.local import LocalDatastore
from monailabel.datastore.utils.convert import binary_to_image, dicom_to_nifti, nifti_to_dicom_seg
from monailabel.datastore.utils.dicom import dicom_web_download_series, dicom_web_upload_dcm
from monailabel.interfaces.datastore import DefaultLabelTag

logger = logging.getLogger(__name__)


class DICOMwebClientX(DICOMwebClient):
    def _decode_multipart_message(self, response: requests.Response, stream: bool) -> Iterator[bytes]:
        content_type = response.headers["content-type"]
        media_type, *ct_info = (ct.strip() for ct in content_type.split(";"))
        if media_type.lower() != "multipart/related":
            response.headers["content-type"] = "multipart/related"
        return super()._decode_multipart_message(response, stream)  # type: ignore


class DICOMWebDatastore(LocalDatastore):
    def __init__(
        self,
        client: DICOMwebClient,
        search_filter: Dict[str, Any],
        cache_path: Optional[str] = None,
        fetch_by_frame=False,
        convert_to_nifti=True,
    ):
        self._client = client
        self._search_filter = search_filter
        self._fetch_by_frame = fetch_by_frame
        self._convert_to_nifti = convert_to_nifti

        uri_hash = hashlib.md5(self._client.base_url.encode("utf-8")).hexdigest()
        datastore_path = (
            os.path.join(cache_path, uri_hash)
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "dicom", uri_hash)
        )
        logger.info(f"DICOMWeb Datastore (cache) Path: {datastore_path}; FetchByFrame: {fetch_by_frame}")
        logger.info(f"DICOMWeb Convert To Nifti: {convert_to_nifti}")
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

        if not self._convert_to_nifti:
            return image_dir

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

        if not self._convert_to_nifti:
            return label_dir

        label_nii_gz = os.path.realpath(
            os.path.join(self._datastore.label_path(DefaultLabelTag.FINAL), f"{image_id}.nii.gz")
        )
        if not os.path.exists(label_nii_gz):
            label_nii_gz = dicom_to_nifti(label_dir, is_seg=True)
            if label_nii_gz:
                super().save_label(image_id, label_nii_gz, label_tag, self._dicom_info(label_id))

        return label_nii_gz

    def _dicom_info(self, series_id):
        meta = Dataset.from_json(self._client.search_for_series(search_filters={"SeriesInstanceUID": series_id})[0])
        fields = ["StudyDate", "StudyTime", "Modality", "RetrieveURL", "PatientID", "StudyInstanceUID"]

        info = {"SeriesInstanceUID": series_id}
        for f in fields:
            info[f] = str(meta[f].value) if meta.get(f) else "UNK"
        return info

    @cached(cache=TTLCache(maxsize=16, ttl=settings.MONAI_LABEL_DICOMWEB_CACHE_EXPIRY))
    def list_images(self) -> List[str]:
        datasets = self._client.search_for_series(search_filters=self._search_filter)
        series = [str(Dataset.from_json(ds)["SeriesInstanceUID"].value) for ds in datasets]
        logger.debug("Total Series: {}\n{}".format(len(series), "\n".join(series)))
        return series

    @cached(cache=TTLCache(maxsize=16, ttl=settings.MONAI_LABEL_DICOMWEB_CACHE_EXPIRY))
    def get_labeled_images(self) -> List[str]:
        datasets = self._client.search_for_series(search_filters={"Modality": "SEG"})
        all_segs = [Dataset.from_json(ds) for ds in datasets]

        image_series = []
        for seg in all_segs:
            meta = self._client.retrieve_series_metadata(
                str(seg["StudyInstanceUID"].value), str(seg["SeriesInstanceUID"].value)
            )
            seg_meta = Dataset.from_json(meta[0])
            if seg_meta.get("ReferencedSeriesSequence"):
                referenced_series_instance_uid = str(
                    seg_meta["ReferencedSeriesSequence"].value[0]["SeriesInstanceUID"].value
                )
                if referenced_series_instance_uid in self.list_images():
                    image_series.append(referenced_series_instance_uid)
                else:
                    logger.warning(
                        "Label Ignored:: ReferencedSeriesSequence is NOT in filtered image list: {}".format(
                            str(seg["SeriesInstanceUID"].value)
                        )
                    )
            else:
                logger.warning(
                    "Label Ignored:: ReferencedSeriesSequence is NOT found: {}".format(
                        str(seg["SeriesInstanceUID"].value)
                    )
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
            image_info = self.get_image_info(image_id)
            label_info.update(
                {
                    "SeriesInstanceUID": label_series_id,
                    "Modality": image_info.get("Modality"),
                    "PatientID": image_info.get("PatientID"),
                    "StudyInstanceUID": image_info.get("StudyInstanceUID"),
                }
            )
            os.unlink(label_file)

        label_id = super().save_label(image_id, label_filename, label_tag, label_info)
        logger.info("Save completed!")

        if output_file:
            os.unlink(output_file)
        return label_id

    def _download_labeled_data(self):
        datasets = self._client.search_for_series(search_filters={"Modality": "SEG"})
        all_segs = [Dataset.from_json(ds) for ds in datasets]

        image_labels = []
        for seg in all_segs:
            meta = self._client.retrieve_series_metadata(
                str(seg["StudyInstanceUID"].value), str(seg["SeriesInstanceUID"].value)
            )
            seg_meta = Dataset.from_json(meta[0])
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
