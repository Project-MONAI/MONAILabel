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

import datetime
import json
import logging
import os
from urllib.parse import quote_plus

import requests
from requests.structures import CaseInsensitiveDict

"""
MonaiServerREST provides the REST endpoints to the MONAIServer
"""


class MonaiServerREST:
    def __init__(self, serverUrl: str):
        self.PARAMS_PREFIX_REST_REQUEST = "params"
        self.serverUrl = serverUrl

    def getServerUrl(self) -> str:
        return self.serverUrl

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()

    def requestDataStoreInfo(self) -> dict:
        download_uri = f"{self.serverUrl}/datastore/?output=all"

        try:
            response = requests.get(download_uri, timeout=5)
        except Exception as exception:
            logging.warning(f"{self.getCurrentTime()}: Request for DataStoreInfo failed due to '{exception}'")
            return None
        if response.status_code != 200:
            logging.warning(
                "{}: Request for datastore-info failed (url: '{}'). Response code is {}".format(
                    self.getCurrentTime(), download_uri, response.status_code
                )
            )
            return None

        return response.json()

    def getDicomDownloadUri(self, image_id: str) -> str:
        download_uri = f"{self.serverUrl}/datastore/image?image={quote_plus(image_id)}"
        logging.info(f"{self.getCurrentTime()}: REST: request dicom image '{download_uri}'")
        return download_uri

    def requestSegmentation(self, image_id: str, tag: str) -> requests.models.Response:
        if tag == "":
            tag = "final"
        download_uri = f"{self.serverUrl}/datastore/label?label={quote_plus(image_id)}&tag={quote_plus(tag)}"
        logging.info(f"{self.getCurrentTime()}: REST: request segmentation '{download_uri}'")

        try:
            response = requests.get(download_uri, timeout=5)
        except Exception as exception:
            logging.warning(
                "{}: Segmentation request (image id: '{}') failed due to '{}'".format(
                    self.getCurrentTime(), image_id, exception
                )
            )
            return None
        if response.status_code != 200:
            logging.warn(
                "{}: Segmentation request (image id: '{}') failed due to response code: '{}'".format(
                    self.getCurrentTime(), image_id, response.status_code
                )
            )
            return None

        return response

    def checkServerConnection(self) -> bool:
        if not self.serverUrl:
            self.serverUrl = "http://127.0.0.1:8000"
        url = self.serverUrl.rstrip("/")

        try:
            response = requests.get(url, timeout=5)
        except Exception as exception:
            logging.warning(f"{self.getCurrentTime()}: Connection to Monai Server failed due to '{exception}'")
            return False
        if response.status_code != 200:
            logging.warn(
                "{}: Server connection Failed. (response code = {}) ".format(
                    self.getCurrentTime(), response.status_code
                )
            )
            return False

        logging.info(f"{self.getCurrentTime()}: Successfully connected to server (server url: '{url}').")
        return True

    def updateLabelInfo(self, image_id: str, tag: str, params: dict) -> int:
        """
        the image_id is the unique ID of an radiographic image
        If the image has a label/segmentation, its label/label_id corresponds to its image_id
        """
        embeddedParams = self.embeddedLabelContentInParams(params)
        logging.info(f"Sending updated label info: {embeddedParams}")

        url = f"{self.serverUrl}/datastore/updatelabelinfo?label={quote_plus(image_id)}&tag={quote_plus(tag)}"
        headers = CaseInsensitiveDict()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        headers["accept"] = "application/json"

        try:
            response = requests.put(url, headers=headers, data=embeddedParams)
        except Exception as exception:
            logging.warning(
                "{}: Update meta data (image id: '{}') failed due to '{}'".format(
                    self.getCurrentTime(), image_id, exception
                )
            )
            return None
        if response.status_code != 200:
            logging.warn(
                "{}: Update meta data (image id: '{}') failed due to response code = {}) ".format(
                    self.getCurrentTime(), image_id, response.status_code
                )
            )
            return response.status_code

        logging.info(f"{self.getCurrentTime()}: Meta data was updated successfully (image id: '{image_id}').")
        return response.status_code

    def embeddedLabelContentInParams(self, labelContent: dict) -> dict:
        params = {}
        params[self.PARAMS_PREFIX_REST_REQUEST] = json.dumps(labelContent)
        return params

    def saveLabel(self, imageId: str, labelDirectory: str, tag: str, params: dict):
        if params is not None:
            embeddedParams = self.embeddedLabelContentInParams(params)
        logging.info(f"{self.getCurrentTime()}: Label and Meta data (image id: '{imageId}'): '{embeddedParams}'")

        url = f"{self.serverUrl}/datastore/label?image={imageId}"
        if tag:
            url += f"&tag={tag}"

        try:
            with open(os.path.abspath(labelDirectory), "rb") as f:
                response = requests.put(url, data=embeddedParams, files={"label": (imageId + ".nrrd", f)})

        except Exception as exception:
            logging.error(
                "{}: Label and Meta data update failed (image id: '{}', meta data: '{}', due to '{}'".format(
                    self.getCurrentTime(), imageId, embeddedParams, exception
                )
            )

        if response.status_code == 200:
            logging.info(
                f"{self.getCurrentTime()}: Label and Meta data was updated successfully (image id: '{imageId}')."
            )
            logging.warn(f"{self.getCurrentTime()}: Meta : '{embeddedParams}'")
        else:
            logging.warn(
                "{}: Update label (image id: '{}') failed due to response code = {}) ".format(
                    self.getCurrentTime(), imageId, response.status_code
                )
            )
        return response.status_code

    def deleteLabelByVersionTag(self, imageId: str, versionTag: str) -> int:
        url = f"{self.serverUrl}/datastore/label?id={imageId}&tag={versionTag}"
        try:
            response = requests.delete(url)
        except Exception as exception:
            logging.error(
                "{}: Label and Meta data deletion failed (image id: '{}', version tag: '{}') due to '{}'".format(
                    self.getCurrentTime(), imageId, versionTag, exception
                )
            )

        if response.status_code == 200:
            logging.info(
                f"{self.getCurrentTime()}: Label and Meta data was deleted successfully (image id: '{imageId}') | tae: '{versionTag}'."
            )
        else:
            logging.warn(
                "{}: Deletion of label (image id: '{}') failed due to response code = {}) ".format(
                    self.getCurrentTime(), imageId, response.status_code
                )
            )
        return response.status_code
