import datetime
import logging
import os
import tempfile
from typing import Dict, List

import requests
import SampleData
import slicer
from ReviewerLibs.ImageData import ImageData
from ReviewerLibs.ImageDataController import ImageDataController
from slicer.ScriptedLoadableModule import *


class MONAILabelReviewerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.temp_dir = None

        self.imageDataController: ImageDataController = ImageDataController()

    # Section: Server
    def getServerUrl(self) -> str:
        return self.imageDataController.getServerUrl()

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()

    def connectToMonaiServer(self, serverUrl: str) -> bool:
        return self.imageDataController.connectToMonaiServer(serverUrl)

    def getMapIdToImageData(self) -> Dict[str, ImageData]:
        """
        Returns dictionary (Dict[str:ImageData]) which maps id to Imagedata-object
        """
        return self.imageDataController.getMapIdToImageData()

    def initMetaDataProcessing(self) -> bool:
        return self.imageDataController.initMetaDataProcessing()

    def getStatistics(self) -> dict:
        return self.imageDataController.getStatistics()

    def getClientIds(self) -> List[str]:
        return self.imageDataController.getClientIds()

    def getReviewers(self) -> List[str]:
        return self.imageDataController.getReviewers()

    # Section: Loading images
    def getAllImageData(self, segmented, isNotSegmented, isApproved, isFlagged) -> List[ImageData]:
        return self.imageDataController.getAllImageData(segmented, isNotSegmented, isApproved, isFlagged)

    def getImageDataByClientId(self, selectedClientId, isApproved, isFlagged) -> List[ImageData]:
        return self.imageDataController.getImageDataByClientId(selectedClientId, isApproved, isFlagged)

    def getPercentageApproved(self, selectedClientId):
        percentageApprovedOfClient, idxApprovedOfClient = self.imageDataController.getPercentageApproved(
            selectedClientId
        )
        return percentageApprovedOfClient, idxApprovedOfClient

    def getPercentageSemgmentedByClient(self, selectedClientId):
        percentageSemgmentedByClient, idxSegmentedByClient = self.imageDataController.getPercentageSemgmentedByClient(
            selectedClientId
        )
        return percentageSemgmentedByClient, idxSegmentedByClient

    # Section: Search Image
    def getMultImageDataByIds(self, idList) -> Dict[str, ImageData]:
        return self.imageDataController.getMultImageDataByIds(idList)

    def searchByAnnotatorReviewer(
        self, selectedAnnotator: str, selectedReviewer: str, isApproved: bool, isFlagged: bool
    ) -> Dict[str, ImageData]:
        return self.imageDataController.searchByAnnotatorReviewer(
            selectedAnnotator, selectedReviewer, isApproved, isFlagged
        )

    def searchByLevel(self, isEasy: bool, isMedium: bool, isHard: bool) -> Dict[str, ImageData]:
        return self.imageDataController.getImageDataByLevel(isEasy=isEasy, isMedium=isMedium, isHard=isHard)

    # Section: Dicom stream
    def updateLabeInfo(self, imageId, updatedMetaJson):
        self.imageDataController.updateLabeInfo(imageId, updatedMetaJson)

    def loadDicomAndSegmentation(self, imageData):
        """
        Loads original Dicom image and Segmentation into slicer window
        Parameters:
          imageData (ImageData): Contains meta data (of dicom and segmenation)
                                 which is required for rest request to monai server
                                 in order to get dicom and segmenation (.nrrd).
        """
        # Request dicom
        image_name = imageData.getFileName()
        image_id = imageData.getName()
        node_name = imageData.getNodeName()
        checksum = imageData.getCheckSum()
        logging.info(
            "{}: Request Data  image_name='{}', node_name='{}', image_id='{}', checksum='{}'".format(
                self.getCurrentTime(), image_name, node_name, image_id, checksum
            )
        )

        self.requestDicomImage(image_id, image_name, node_name, checksum)
        self.setTempFolderDir()

        # Request segmentation
        if imageData.isSegemented():
            segmentationFileName = imageData.getSegmentationFileName()
            img_blob = self.imageDataController.reuqestSegmentation(image_id)
            destination = self.storeSegmentation(img_blob, segmentationFileName, self.temp_dir.name)
            self.displaySegmention(destination)
            os.remove(destination)
            logging.info(f"{self.getCurrentTime()}: Removed file at {destination}")

    def storeSegmentation(
        self, response: requests.models.Response, segmentationFileName: str, tempDirectory: str
    ) -> str:
        """
        stores loaded segmenation temporarily in local directory
        Parameters:
            response (requests.models.Response): contains segmentation data
            image_id (str): image id of segmentation
        """
        segmentation = response.content
        destination = self.getPathToStore(segmentationFileName, tempDirectory)
        with open(destination, "wb") as img_file:
            img_file.write(segmentation)
        logging.info(f"{self.getCurrentTime()}: Images is stored in: {destination}")
        return destination

    def getPathToStore(self, segmentationFileName: str, tempDirectory: str) -> str:
        return tempDirectory + "/" + segmentationFileName

    def displaySegmention(self, destination: str):
        """
        Displays the segmentation in slicer window
        """
        segmentation = slicer.util.loadSegmentation(destination)

    def requestDicomImage(self, image_id: str, image_name: str, node_name: str, checksum: str):
        download_uri = self.imageDataController.getDicomDownloadUri(image_id)
        sampleDataLogic = SampleData.SampleDataLogic()
        _volumeNode = sampleDataLogic.downloadFromURL(
            nodeNames=node_name, fileNames=image_name, uris=download_uri, checksums=checksum
        )[0]

    def setTempFolderDir(self):
        """
        Create temporary dirctory to store the downloaded segmentation (.nrrd)
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory()
        logging.info(f"{self.getCurrentTime()}: Temporary Directory: '{self.temp_dir.name}'")
