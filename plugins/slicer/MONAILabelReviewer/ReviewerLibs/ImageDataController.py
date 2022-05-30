import datetime
import logging
from typing import Dict, List

import requests
from ReviewerLibs.ImageData import ImageData
from ReviewerLibs.ImageDataExtractor import ImageDataExtractor
from ReviewerLibs.JsonParser import JsonParser
from ReviewerLibs.MonaiServerREST import MonaiServerREST

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

    def getServerUrl(self) -> str:
        return self.monaiServerREST.getServerUrl()

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()

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

    def getStatistics(self) -> dict:
        """
        returns a map which contains statistical values which are comming from ImageDataExtractor object
        """
        statistics = {}
        # ProgressBar: TOTAL
        statistics["segmentationProgress"] = self.imageDataExtractor.getSegmentationProgessInPercentage()
        statistics["idxTotalSegmented"] = self.imageDataExtractor.getSegmentationVsTotalStr()
        statistics["idxTotalApproved"] = self.imageDataExtractor.getApprovalVsTotal()
        statistics["progressPercentage"] = self.imageDataExtractor.getApprovalProgressInPercentage()

        # ProgressBar: FILTER (incl. idxTotalSegmented, idxTotalApproved)
        statistics["segmentationProgressAllPercentage"] = self.imageDataExtractor.getSegmentationProgessInPercentage()
        statistics["approvalProgressPercentage"] = self.imageDataExtractor.getApprovalProgressInPercentage()

        return statistics

    # returns only client id of those images which are segemented
    def getClientIds(self) -> List[str]:
        return self.imageDataExtractor.getClientIds()

    def getReviewers(self) -> List[str]:
        return self.imageDataExtractor.getReviewers()

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

    # Section: Dicom stream
    def updateLabeInfo(self, imageId, updatedMetaJson) -> bool:
        """
        sends meta information via http request to monai server
        in order to perist the information in datastore_v2.json file

        returns True if successfully sent http request
        else False
        """
        repsonseCode = self.monaiServerREST.updateLabeInfo(image_id=imageId, params=updatedMetaJson)
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

    def reuqestSegmentation(self, image_id) -> requests.models.Response:
        """
        after sending request to monai server
        rerturns response body (img_blob) which contains the segmentation data
        """
        img_blob = self.monaiServerREST.requestSegmentation(image_id)
        logging.info(
            "{}: Segmentation successfully requested from MONAIServer (image id: {})".format(
                self.getCurrentTime(), image_id
            )
        )
        return img_blob

    def getDicomDownloadUri(self, image_id: str) -> str:
        return self.monaiServerREST.getDicomDownloadUri(image_id)
