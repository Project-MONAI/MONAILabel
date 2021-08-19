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

import json
import logging
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Dict, List, Optional

import pydicom
import requests
from dicomweb_client.api import DICOMwebClient

from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.utils.datastore.dicom.attributes import (
    ATTRB_MODALITY,
    ATTRB_MONAILABELINFO,
    ATTRB_MONAILABELTAG,
    ATTRB_PATIENTID,
    ATTRB_REFERENCEDSERIESSEQUENCE,
    ATTRB_SERIESINSTANCEUID,
    ATTRB_SOPINSTANCEUID,
    ATTRB_STUDYINSTANCEUID,
    DICOMSEG_MODALITY,
)
from monailabel.utils.datastore.dicom.datamodel import DICOMObjectModel
from monailabel.utils.datastore.dicom.util import generate_key

logger = logging.getLogger(__name__)


class DICOMWebClient(DICOMwebClient):
    def __init__(
        self,
        url: str,
        session: Optional[requests.Session] = None,
        qido_url_prefix: Optional[str] = None,
        wado_url_prefix: Optional[str] = None,
        stow_url_prefix: Optional[str] = None,
        delete_url_prefix: Optional[str] = None,
        proxies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        callback: Optional[Callable] = None,
        chunk_size: int = 10 ** 9,
        num_download_threads: int = 8,
    ) -> None:
        super().__init__(
            url,
            session=session,
            qido_url_prefix=qido_url_prefix,
            wado_url_prefix=wado_url_prefix,
            stow_url_prefix=stow_url_prefix,
            delete_url_prefix=delete_url_prefix,
            proxies=proxies,
            headers=headers,
            callback=callback,
            chunk_size=chunk_size,
        )
        self.num_download_threads = num_download_threads

    def retrieve_dataset(self) -> Dict[str, DICOMObjectModel]:

        series = self.search_for_series()
        objects: Dict[str, DICOMObjectModel] = {}

        for s in series:
            s_patient_id = s[ATTRB_PATIENTID]["Value"][0]
            s_study_id = s[ATTRB_STUDYINSTANCEUID]["Value"][0]
            s_series_id = s[ATTRB_SERIESINSTANCEUID]["Value"][0]
            key = generate_key(s_patient_id, s_study_id, s_series_id)

            s_meta = self.retrieve_series_metadata(
                study_instance_uid=s_study_id,
                series_instance_uid=s_series_id,
            )

            s_tag: str = DefaultLabelTag.FINAL.value
            if s.get(ATTRB_MONAILABELTAG):
                s_tag = s_meta[0][ATTRB_MONAILABELTAG]["Value"][0]

            s_info: Dict = {}
            if s.get(ATTRB_MONAILABELINFO):
                s_info = json.loads(s_meta[0][ATTRB_MONAILABELINFO]["Value"][0])

            # determine if this is a DICOMSEG series
            if s[ATTRB_MODALITY]["Value"][0] == DICOMSEG_MODALITY:

                s_info.update({"object_type": "label"})

                # add DICOMSEG to datastore
                objects.update(
                    {
                        key: DICOMObjectModel(
                            patient_id=s_patient_id,
                            study_id=s_study_id,
                            series_id=s_series_id,
                            tag=s_tag,
                            info=s_info,
                        )
                    }
                )

            else:  # this is an original image

                # find all DICOMSEG labels related to this image first
                related_labels_keys = []
                for label in series:
                    label_patient_id = label[ATTRB_PATIENTID]["Value"][0]
                    label_study_id = label[ATTRB_STUDYINSTANCEUID]["Value"][0]
                    label_series_id = label[ATTRB_SERIESINSTANCEUID]["Value"][0]

                    if DICOMSEG_MODALITY in label[ATTRB_MODALITY]["Value"] and s_patient_id == label_patient_id:

                        label_series_meta = self.retrieve_series_metadata(
                            study_instance_uid=label_study_id, series_instance_uid=label_series_id
                        )[
                            0
                        ]  # assuming a multiframe DICOMSEG (single-image series)
                        label_referenced_series = []
                        if label_series_meta.get(ATTRB_REFERENCEDSERIESSEQUENCE) and label_series_meta[
                            ATTRB_REFERENCEDSERIESSEQUENCE
                        ]["Value"][0].get(ATTRB_SERIESINSTANCEUID):
                            label_referenced_series.extend(
                                label_series_meta[ATTRB_REFERENCEDSERIESSEQUENCE]["Value"][0][ATTRB_SERIESINSTANCEUID][
                                    "Value"
                                ]
                            )

                        # to find the related original iage of this label we must look at all instances of a label
                        # in the attribute
                        if s_series_id in label_referenced_series:

                            label_key = generate_key(label_patient_id, label_study_id, label_series_id)
                            related_labels_keys.append(label_key)

                s_info.update({"object_type": "image"})

                objects.update(
                    {
                        key: DICOMObjectModel(
                            patient_id=s_patient_id,
                            study_id=s_study_id,
                            series_id=s_series_id,
                            info=s_info,
                            related_labels_keys=related_labels_keys,
                        )
                    }
                )

        return objects

    def get_object_url(self, dicom_object: DICOMObjectModel):
        return self._get_series_url("wado", dicom_object.study_id, dicom_object.series_id).replace(self.base_url, "")

    def get_object(self, dicom_object: DICOMObjectModel) -> List[pydicom.Dataset]:
        series_meta = self.retrieve_series_metadata(dicom_object.study_id, dicom_object.series_id)
        instances: List[pydicom.Dataset] = []

        list_lock = Lock()

        def download_instance(idx):
            instance = self.retrieve_instance(
                dicom_object.study_id,
                dicom_object.series_id,
                series_meta[idx][ATTRB_SOPINSTANCEUID]["Value"][0],
            )
            with list_lock:
                instances.append(instance)

        with ThreadPool(self.num_download_threads) as tp:
            tp.map(download_instance, range(len(series_meta)))

        return instances

    def push_series(self, image_object: DICOMObjectModel, dataset: List[pydicom.Dataset]):
        self.store_instances(
            study_instance_uid=image_object.study_id,
            datasets=dataset,
        )

    # workaround for Orthanc compatibility of DICOMWeb headers where
    # Orthanc doesn't support multiple `Content-type`s
    def _http_get_application_json(
        self, url: str, params: Optional[Dict[str, Any]] = None, stream: bool = False
    ) -> List[Dict[str, dict]]:
        """Performs a HTTP GET request that accepts "applicaton/dicom+json"
        or "application/json" media type.

        Parameters
        ----------
        url: str
            unique resource locator
        params: Dict[str], optional
            query parameters
        stream: bool, optional
            whether data should be streamed (i.e., requested using chunked
            transfer encoding)

        Returns
        -------
        List[str, dict]
            content of HTTP message body in DICOM JSON format

        """
        content_type = "application/dicom+json"
        response = self._http_get(url, params=params, headers={"Accept": content_type}, stream=stream)
        if response.content:
            decoded_response: List[Dict[str, dict]] = response.json()
            # All metadata resources are expected to be sent as a JSON array of
            # DICOM data sets. However, some origin servers may incorrectly
            # sent an individual data set.
            if isinstance(decoded_response, dict):
                return [decoded_response]
            return decoded_response
        return []
