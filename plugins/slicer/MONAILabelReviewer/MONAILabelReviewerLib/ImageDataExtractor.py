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

from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.MONAILabelReviewerEnum import Level

"""
ImageDataExtractor gets dictionary (mapping from id to ImageData from JsonParser) and caches
    Mapping:
        - imageIds TO ImageData,
        - client TO list of imageIds
    List:
        - imageIds of all images which are not segemented yet
        - imageIds of all images which are approved
        - all reviewers

Each modification during review process will be stored in corresponding ImageData
ImageDataExtractor provides the meta data across all ImageData-Containers when the user selects the filter option
"""


class ImageDataExtractor:
    def __init__(self, nameToImageData: dict):
        self.LEVEL = Level()
        self.nameToImageData: Dict[str, ImageData] = nameToImageData

        self.clientToImageIds: Dict[str, list] = {}
        self.idsOfNotSegmented: List[str] = []
        self.idsOfApprovedSementations: List[str] = []
        self.reviewers: List[str] = []

    def init(self):
        self.groupImageDataByClientId()
        self.extractAllReviewers()
        self.extractNotSegmentedImageIds()

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()

    def groupImageDataByClientId(self):
        for imageId, imageData in self.nameToImageData.items():
            if imageData.isSegemented():
                clientId = imageData.getClientId()
                if clientId:
                    if clientId not in self.clientToImageIds:
                        self.clientToImageIds[clientId] = []
                    self.clientToImageIds[clientId].append(imageId)

    def extractAllReviewers(self):
        for imageData in self.nameToImageData.values():
            if imageData.isSegemented():
                reviewer = imageData.getApprovedBy()
                if reviewer not in self.reviewers and reviewer != "":
                    self.reviewers.append(reviewer)

    def extractNotSegmentedImageIds(self):
        for imageId, imageData in self.nameToImageData.items():
            if imageData.isSegemented() is False:
                self.idsOfNotSegmented.append(imageId)

    def getTotalNumImages(self) -> int:
        return len(self.nameToImageData)

    def getImageDataIds(self) -> List[str]:
        return [*self.nameToImageData.keys()]

    def getClientIds(self) -> List[str]:
        return [*self.clientToImageIds.keys()]

    def getReviewers(self) -> List[str]:
        return self.reviewers

    def getImageDataNotsegmented(self) -> List[ImageData]:
        """
        returns list of ImageData of corresponingd image studies wich has not been segemeted
        """
        notSegmented = []
        for id in self.idsOfNotSegmented:
            imageData = self.nameToImageData[id]
            notSegmented.append(imageData)
        return notSegmented

    def getNumOfNotSegmented(self) -> int:
        return len(self.idsOfNotSegmented)

    def getNumOfSegmented(self) -> int:
        count = 0
        for idList in self.clientToImageIds.values():
            count += len(idList)
        return count

    def getSegmentationProgessInPercentage(self) -> int:
        """
        returns percentage of already segmented images out of all available images
        """
        segmentedCount = self.getNumOfSegmented()
        float_Num = segmentedCount / self.getTotalNumImages()
        return int(float_Num * 100)

    def getSegmentationVsTotalStr(self) -> str:
        """
        returns the index of subjected imageData within imageData data set
        """
        segmentedCount = self.getNumOfSegmented()
        idxTotalSegmented = f"{segmentedCount}/{self.getTotalNumImages()}"
        return idxTotalSegmented

    def getApprovalProgressInPercentage(self) -> int:
        """
        returns percentage of already approved imageData out of all available imageData
        """
        approvalCount = self.getNumApprovedSegmentation()
        fraction = approvalCount / self.getTotalNumImages()
        return int(fraction * 100)

    def getApprovalVsTotal(self) -> str:
        approvalCount = self.getNumApprovedSegmentation()
        idxTotalApproved = f"{approvalCount}/{self.getTotalNumImages()}"
        return idxTotalApproved

    def invalidFilterCombination(self, segmented: bool, notSegmented: bool, approved: bool, flagged: bool) -> bool:
        return (
            (notSegmented is True and segmented is True)
            or (approved is True and flagged is True)
            or (notSegmented is True and approved is True)
            or (notSegmented is True and flagged is True)
        )

    def getAllImageData(self, segmented=False, notSegmented=False, approved=False, flagged=False) -> List[ImageData]:
        """
        returns fitered list of imageData which are filtered according to input parameters
        """
        if self.invalidFilterCombination(segmented, notSegmented, approved, flagged):
            logging.warning(
                "{}: Selected filter options are not valid: segmented='{}' | notSegmented='{}' | approved='{}' | flagged='{}')".format(
                    self.getCurrentTime(), segmented, notSegmented, approved, flagged
                )
            )
            return None

        if notSegmented is False and segmented is False and approved is False and flagged is False:
            return [*self.nameToImageData.values()]

        selectedImageData = []
        for imagedata in self.nameToImageData.values():

            if notSegmented is True and segmented is False and imagedata.isSegemented() is False:
                selectedImageData.append(imagedata)
                continue

            if imagedata.isSegemented() is segmented and imagedata.isApproved() is True and approved is True:
                selectedImageData.append(imagedata)
                continue

            if (
                imagedata.isSegemented() is segmented
                # and imagedata.isApproved() is approved
                and imagedata.isFlagged() is True
                and flagged is True
            ):
                selectedImageData.append(imagedata)
                continue

        return selectedImageData

    def getImageDataByClientId(self, clientId: str, approved=False, flagged=False) -> List[ImageData]:
        """
        returns fitered list of imageData which are filtered according to client (=annotator) and parameters (approved, flagged)
        """
        if clientId == "":
            return None
        if approved and flagged:
            logging.warning(
                "{}: Selected filter options are not valid: approved='{}' and flagged='{}')".format(
                    self.getCurrentTime(), approved, flagged
                )
            )
            return None

        imageIds = self.clientToImageIds[clientId]

        if approved is False and flagged is False:
            return self.extractImageDataByIds(imageIds)
        else:
            return self.extractImageDataByApprovedAndFlaggedStatus(clientId, approved, flagged, imageIds)

    def extractImageDataByIds(self, imageIds: List[str]) -> List[ImageData]:
        imageDataList = []
        for id in imageIds:
            imageData = self.nameToImageData[id]
            imageDataList.append(imageData)
        return imageDataList

    def extractImageDataByApprovedAndFlaggedStatus(
        self, clientId: str, approved: bool, flagged: bool, imageIds: List[str]
    ) -> List[ImageData]:
        imageDataList = []
        for id in imageIds:
            if id not in self.nameToImageData:
                logging.error(
                    "{}: Image data [id = {}] not found for [clientId = {}] ".format(
                        self.getCurrentTime(), id, clientId
                    )
                )
                continue
            imageData = self.nameToImageData[id]
            if imageData.hasSegmentationMeta() is False:
                continue
            if approved and imageData.isApproved() is False:
                continue
            if flagged and imageData.isFlagged() is False:
                continue

            imageDataList.append(imageData)
        return imageDataList

    def getImageDataByClientAndReviewer(
        self, clientId: str, reviewerId: str, approved=False, flagged=False
    ) -> List[ImageData]:
        """
        returns fitered list of imageData which are filtered according to client (=annotator) and reviewer and parameters (approved, flagged)
        """

        imageDatas = self.getImageDataByClientId(clientId, approved, flagged)
        filteredByRewiewer = list(filter(lambda imageData: (imageData.getApprovedBy() == reviewerId), imageDatas))
        return filteredByRewiewer

    def getImageDataByReviewer(self, reviewerId: str, approved=False, flagged=False) -> List[ImageData]:
        if reviewerId == "":
            return None
        if approved and flagged:
            logging.warning(
                "{}: Selected filter options are not valid: approved='{}' and flagged='{}')".format(
                    self.getCurrentTime(), approved, flagged
                )
            )
            return None

        filteredImageDataList = []

        for imageData in self.nameToImageData.values():

            if imageData.isSegemented() is False:
                continue
            if approved and imageData.isApproved() is False:
                continue
            if flagged and imageData.isFlagged() is False:
                continue
            if imageData.getApprovedBy() == reviewerId:
                filteredImageDataList.append(imageData)

        return filteredImageDataList

    def getImageDataByLevel(self, isEasy: bool, isMedium: bool, isHard: bool) -> Dict[str, ImageData]:
        """
        returns fitered list of imageData which are filtered according to level of difficulty (regarding segmentation): easy, medium, hard
        """
        filteredImageData = {}
        for id, imagedata in self.nameToImageData.items():
            if imagedata is None:
                continue
            if imagedata.isSegemented() is False:
                continue
            if isEasy and imagedata.getLevel() == self.LEVEL.EASY:
                filteredImageData[id] = imagedata
                continue

            if isMedium and imagedata.getLevel() == self.LEVEL.MEDIUM:
                filteredImageData[id] = imagedata
                continue

            if isHard and imagedata.getLevel() == self.LEVEL.HARD:
                filteredImageData[id] = imagedata
        return filteredImageData

    def getSingleImageDataById(self, imageId: str) -> ImageData:
        """
        returns imageData by given imageId
        """
        if self.isBlank(imageId):
            return None
        if imageId not in self.nameToImageData:
            logging.warning(f"{self.getCurrentTime()}: Image data for requested id [{imageId}] not found")
            return None
        return self.nameToImageData[imageId]

    def getMultImageDataByIds(self, ids: List[str]) -> Dict[str, ImageData]:
        """
        returns multiple imageData by given list of imageId
        """
        idToimageData: Dict[str, ImageData] = {}
        if len(ids) == 0:
            logging.warning(f"{self.getCurrentTime()}: Given id list is empty.")
            return {}
        for id in ids:
            imageData = self.getSingleImageDataById(id)
            if imageData is None:
                continue
            idToimageData[imageData.getName()] = imageData
        return idToimageData

    def getNumApprovedSegmentation(self) -> int:
        """
        returns total number of imageData which are approved
        """
        count = self.countApprovedSegmentation(self.nameToImageData.values())
        return count

    def countApprovedSegmentation(self, imageDatas: List[ImageData]) -> int:
        if imageDatas is None:
            return 0
        approvedCount = 0
        for imageData in imageDatas:
            if imageData is None:
                continue
            if imageData.isApproved():
                approvedCount += 1
        return approvedCount

    def getPercentageApproved(self, clientId: str):
        """
        returns the percentage of images that have already been approved by given client (=Annotator)
        and the value: (total number of images approved by given client (=Annotator))/(total number of imageData)
        """
        listImageData = self.getImageDataByClientId(clientId=clientId)
        approvedCount = self.countApprovedSegmentation(listImageData)
        if len(listImageData) == 0:
            logging.warning(f"{self.getCurrentTime()}: There are no images")
            return 0
        fraction = approvedCount / len(listImageData)
        precentage = int(fraction * 100)
        idxApprovedOfClient: str = f"{approvedCount}/{len(listImageData)}"
        return precentage, idxApprovedOfClient

    def getPercentageSemgmentedByClient(self, clientId: str):
        """
        returns the percentage of images that have already been segmented by given client (=Annotator)
        and the value: (total number of images segmented by given client (=Annotator))/(total number of imageData)
        """
        numSegementedByClient = len(self.clientToImageIds[clientId])
        fraction = numSegementedByClient / self.getTotalNumImages()
        precentage = int(fraction * 100)
        idxSegmentedByClient: str = f"{numSegementedByClient}/{self.getTotalNumImages()}"
        return precentage, idxSegmentedByClient

    def getApprovedSegmentationIds(self) -> List[str]:
        """
        returns list of ids of all approved imageData
        """
        idsOfApprovedSementations = []
        for imageId, imageData in self.nameToImageData.items():
            if imageData.isApproved():
                idsOfApprovedSementations.append(imageId)
        return idsOfApprovedSementations

    def getSegmentedImageIds(self) -> List[str]:
        """
        returns list of ids of all segmented imageData
        """
        idsOfSegmented = []
        for imageId, imageData in self.nameToImageData.items():
            if imageData.isSegemented():
                idsOfSegmented.append(imageId)
        return idsOfSegmented

    def isBlank(self, string) -> bool:
        return not (string and string.strip())
