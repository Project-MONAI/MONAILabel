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
import logging
from typing import Dict, List

import requests
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.ImageDataExtractor import ImageDataExtractor
from MONAILabelReviewerLib.ImageDataStatistics import ImageDataStatistics
from MONAILabelReviewerLib.JsonParser import JsonParser
from MONAILabelReviewerLib.MonaiServerREST import MonaiServerREST

"""
ImageDataController manages all data processing and data transactions via

    1. MonaiServerREST (requests and peristency of meta data, image data, segmentation data from monai server)
    2. ImageDataExtractor (handling of logical operations coming from set of imageData)
    3. JsonParser (parsing information from datastore_v2.json)
    4. ImageData (container which caches information of single image and corresponding segmenation information and meta data)

content of meta data:
    1. "status" (flagged or approved)
    2. "approvedBy" (name of reviewer)
    3. "level" (level of difficulty of segmentation: easy, medium, hard)
    4. "comment" (any comment on image and segmenation)
    5. "editTime"

    list of meta information can be extanded

"""


class ImageDataController:
    def __init__(self):
        self.monaiServerREST: MonaiServerREST = None
        self.imageDataExtractor: ImageDataExtractor = None
        self.temp_dir = None

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()

    # ImageDataExtractor methods

    def initMetaDataProcessing(self) -> bool:
        """
        Passes mapIdToImageData to ImageDataExtractor object in order to process the meta information
        for each imageData

        returns True if it was successful
        else False
        """
        mapIdToImageData = self.getMapIdToImageData()
        if mapIdToImageData is None:
            return False
        self.imageDataExtractor = ImageDataExtractor(mapIdToImageData)
        self.imageDataExtractor.init()
        return True

    # returns only client id of those images which are segemented
    def getClientIds(self) -> List[str]:
        return self.imageDataExtractor.getClientIds()

    def getReviewers(self) -> List[str]:
        return self.imageDataExtractor.getReviewers()

    def getStatistics(self) -> ImageDataStatistics:
        """
        returns a map which contains statistical values which are comming from ImageDataExtractor object
        """
        statistics = ImageDataStatistics()

        statistics.build(
            segmentationProgress=self.imageDataExtractor.getSegmentationProgessInPercentage(),
            idxTotalSegmented=self.imageDataExtractor.getSegmentationVsTotalStr(),
            idxTotalApproved=self.imageDataExtractor.getApprovalVsTotal(),
            progressPercentage=self.imageDataExtractor.getApprovalProgressInPercentage(),
            segmentationProgressAllPercentage=self.imageDataExtractor.getSegmentationProgessInPercentage(),
            approvalProgressPercentage=self.imageDataExtractor.getApprovalProgressInPercentage(),
        )

        return statistics

    # Section: Loading images
    def getAllImageData(self, segmented, isNotSegmented, isApproved, isFlagged) -> List[ImageData]:
        return self.imageDataExtractor.getAllImageData(
            segmented=segmented, notSegmented=isNotSegmented, approved=isApproved, flagged=isFlagged
        )

    def getImageDataByClientId(self, selectedClientId, isApproved, isFlagged) -> List[ImageData]:
        return self.imageDataExtractor.getImageDataByClientId(
            clientId=selectedClientId, approved=isApproved, flagged=isFlagged
        )

    def getPercentageApproved(self, selectedClientId):
        percentageApprovedOfClient, idxApprovedOfClient = self.imageDataExtractor.getPercentageApproved(
            selectedClientId
        )
        return percentageApprovedOfClient, idxApprovedOfClient

    def getPercentageSemgmentedByClient(self, selectedClientId):
        percentageSemgmentedByClient, idxSegmentedByClient = self.imageDataExtractor.getPercentageSemgmentedByClient(
            selectedClientId
        )
        return percentageSemgmentedByClient, idxSegmentedByClient

    # Section: Search Image
    def getMultImageDataByIds(self, imageIds) -> Dict[str, ImageData]:
        return self.imageDataExtractor.getMultImageDataByIds(imageIds)

    def searchByAnnotatorReviewer(
        self, selectedAnnotator: str, selectedReviewer: str, isApproved: bool, isFlagged: bool
    ) -> Dict[str, ImageData]:
        """
        returns set of imageData (imageId mapped to ImageData) according to given filter options
        """
        idToImageData: Dict[str, ImageData] = {}
        imageIdsOfAnnotator = None
        if selectedAnnotator == "All" and selectedReviewer != "All":
            imageIdsOfAnnotator = self.imageDataExtractor.getImageDataByReviewer(
                selectedReviewer, isApproved, isFlagged
            )

        if selectedAnnotator != "All" and selectedReviewer == "All":
            imageIdsOfAnnotator = self.imageDataExtractor.getImageDataByClientId(
                selectedAnnotator, isApproved, isFlagged
            )

        if selectedReviewer == "All" and selectedAnnotator == "All":
            imageIdsOfAnnotator = self.imageDataExtractor.getAllImageData(
                segmented=True, notSegmented=False, approved=isApproved, flagged=isFlagged
            )

        if selectedReviewer != "All" and selectedAnnotator != "All":
            imageIdsOfAnnotator = self.imageDataExtractor.getImageDataByClientAndReviewer(
                selectedAnnotator, selectedReviewer, isApproved, isFlagged
            )

        if imageIdsOfAnnotator is None:
            return idToImageData

        for imageData in imageIdsOfAnnotator:
            idToImageData[imageData.getName()] = imageData

        return idToImageData

    def getImageDataByLevel(self, isEasy: bool, isMedium: bool, isHard: bool) -> Dict[str, ImageData]:
        """
        returns set of imageData (imageId mapped to ImageData) according to given level of difficulty
        """
        imageIdsOfAnnotator = self.imageDataExtractor.getImageDataByLevel(
            isEasy=isEasy, isMedium=isMedium, isHard=isHard
        )
        return imageIdsOfAnnotator

    # MONAI server methods

    def getServerUrl(self) -> str:
        return self.monaiServerREST.getServerUrl()

    def setMonaiServer(self, serverUrl: str):
        self.monaiServerREST = MonaiServerREST(serverUrl)

    def connectToMonaiServer(self, serverUrl: str) -> bool:
        self.setMonaiServer(serverUrl)
        return self.monaiServerREST.checkServerConnection()

    def getMapIdToImageData(self) -> Dict[str, ImageData]:
        """
        Returns dictionary (Dict[str:ImageData]) which maps id to Imagedata-object
        """
        jsonObj = self.monaiServerREST.requestDataStoreInfo()
        if jsonObj is None:
            return None

        # Parse json file to ImageData object
        jsonParser = JsonParser(jsonObj)
        jsonParser.init()
        mapIdToImageData = jsonParser.getMapIdToImageData()
        return mapIdToImageData

    # Section: Dicom stream
    def updateLabelInfoOfAllVersionTags(
        self, imageData: ImageData, versionTag: str, level: str, updatedMetaJson: dict
    ) -> bool:
        imageId = imageData.getName()
        self.updateLabelInfo(imageId, versionTag, updatedMetaJson)

        tagToSegmentationMetaJson = imageData.updateApprovedStatusOfOtherThanSubjectedVersion(
            subjectedTag=versionTag, difficultyLevel=level
        )
        for tag, segmentationMetaJson in tagToSegmentationMetaJson.items():
            self.updateLabelInfo(imageId, tag, segmentationMetaJson)

    def updateLabelInfo(self, imageId, versionTag, updatedMetaJson) -> bool:
        """
        sends meta information via http request to monai server
        in order to perist the information in datastore_v2.json file

        returns True if successfully sent http request
        else False
        """
        repsonseCode = self.monaiServerREST.updateLabelInfo(image_id=imageId, tag=versionTag, params=updatedMetaJson)
        if repsonseCode == 200:
            logging.info(f"{self.getCurrentTime()}: Successfully persist meta data for image (id='{imageId}')")
            return True
        else:
            logging.info(
                "{}: Failed meta date persistence for image (id='{}', response code = '{}')".format(
                    self.getCurrentTime(), imageId, repsonseCode
                )
            )
            return False

    def reuqestSegmentation(self, image_id: str, tag: str) -> requests.models.Response:
        """
        after sending request to monai server
        rerturns response body (img_blob) which contains the segmentation data
        """
        img_blob = self.monaiServerREST.requestSegmentation(image_id, tag)
        logging.info(
            "{}: Segmentation successfully requested from MONAIServer (image id: {})".format(
                self.getCurrentTime(), image_id
            )
        )
        return img_blob

    def getDicomDownloadUri(self, image_id: str) -> str:
        return self.monaiServerREST.getDicomDownloadUri(image_id)

    def saveLabelInMonaiServer(self, image_in: str, label_in: str, tag: str, params: Dict):
        self.monaiServerREST.saveLabel(image_in, label_in, tag, params)

    def deleteLabelByVersionTag(self, imageId: str, versionTag: str) -> bool:
        reponseCode = self.monaiServerREST.deleteLabelByVersionTag(imageId, versionTag)
        if reponseCode == 200:
            return True
        return False
