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

import logging
from datetime import datetime
from typing import Dict, List

from MONAILabelReviewerLib.MONAILabelReviewerEnum import SegStatus
from MONAILabelReviewerLib.SegmentationMeta import SegmentationMeta

"""
ImageData is a container for each segmentation/image.
Such ImageData contains the meta data of corresponding segmentation/image (e.g. fileName, checkSum, comment, etc.)
Each change (regarding the review process) will be monitored within ImageData.
Once a user select the next segmentation during review the information in ImageData will be send to MONAI-Server in order
to persist the data in datastore_v2.json file.

"""


class ImageData:
    def __init__(self, name, fileName, nodeName, checkSum, segmented, timeStamp, comment=""):
        self.name: str = name  # equals imageId
        self.fileName: str = fileName
        self.nodeName: str = nodeName
        self.checkSum: str = checkSum
        self.segmented: bool = segmented
        self.timeStamp: int = timeStamp
        self.comment: str = comment

        self.versionNames: List[str] = []  # equals to labelNames
        self.labelContent: dict = {}
        """
        example of 'labelContent'
            "label_info": [
              {
                "name": "Lung",
                "idx": 1
              },
              {
                "name": "Heart",
                "idx": 2
              },
              {
                "name": "Trachea",
                "idx": 3
              },
              {
                "name": "Mediastinum",
                "idx": 4
              },
              {
                "name": "Clavicle",
                "idx": 5
              }
            ],
        """
        self.segmentationMetaDict: Dict[str, SegmentationMeta] = {}

        self.STATUS = SegStatus()

        self.client_id: str = None
        self.segmentationFileName: str = None
        self.tempDirectory: str = None
        self.prefixVersion = "version_"
        self.FINAL = "final"
        self.ORIGIN = "origin"

    def setVersionNames(self, versionNames: List[str]):
        self.versionNames: List[str] = versionNames

    def setLabelContent(self, labelContent: dict):
        self.labelContent: dict = labelContent

    def setSegmentationMetaDict(self, segmentationMetaDict: Dict[str, SegmentationMeta]):
        self.segmentationMetaDict = segmentationMetaDict

    def getName(self) -> str:
        return self.name

    def getFileName(self) -> str:
        return self.fileName

    def getNodeName(self) -> str:
        return self.nodeName

    def getCheckSum(self) -> str:
        return self.checkSum

    def getsegmentationMetaDict(self) -> dict:
        return self.segmentationMetaDict

    def getClientId(self, versionTag="final") -> str:
        return self.client_id

    def getTimeStamp(self) -> int:
        return self.timeStamp

    def formatTimeStamp(self, timeStamp) -> str:
        if type(timeStamp) == str:
            return timeStamp
        return str(datetime.fromtimestamp(timeStamp))

    def getTimeOfAnnotation(self) -> str:
        return self.formatTimeStamp(self.timeStamp)

    def getTimeOfEditing(self, versionTag="final"):

        if self.isSegemented() is False or self.hasSegmentationMeta(tag=versionTag) is False:
            return ""

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=versionTag)
        if segmentationMeta is None:
            return ""

        formattedTime = self.formatTimeStamp(segmentationMeta.getEditTime())
        return formattedTime

    def isSegemented(self) -> bool:
        return self.segmented

    def getLabelContent(self) -> dict:
        return self.labelContent

    def getComment(self, versionTag="final") -> str:
        if self.isSegemented() is False or self.hasSegmentationMeta(tag=versionTag) is False:
            return ""

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=versionTag)
        if segmentationMeta is None:
            return ""

        return segmentationMeta.getComment()

    def getSegmentationMetaDict(self) -> dict:
        return self.segmentationMetaDict

    def getStatus(self, versionTag="final") -> str:
        if self.isSegemented() is False:
            return self.STATUS.NOT_SEGMENTED

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=versionTag)
        if segmentationMeta is None:
            return ""

        return segmentationMeta.getStatus()

    def getApprovedBy(self, versionTag="final") -> str:
        if self.isSegemented() is False or self.hasSegmentationMeta(tag=versionTag) is False:
            return ""

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=versionTag)
        if segmentationMeta is None:
            return ""

        return segmentationMeta.getApprovedBy()

    def isApprovedVersion(self, versionTag="final") -> bool:
        if self.isSegemented() is False or self.hasSegmentationMeta(tag=versionTag) is False:
            return False

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=versionTag)
        if segmentationMeta is None:
            return False

        status = segmentationMeta.getStatus()
        if status == self.STATUS.APPROVED:
            return True

        return False

    def isApproved(self, versionTag="final") -> bool:
        if self.isSegemented() is False:
            return False

        for segmentationMeta in self.segmentationMetaDict.values():
            status = segmentationMeta.getStatus()
            if status == self.STATUS.APPROVED:
                return True
        return False

    def isFlagged(self, versionTag="final") -> bool:
        if self.isSegemented() is False or self.hasSegmentationMeta(tag=versionTag) is False:
            return False

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=versionTag)
        if segmentationMeta is None:
            return False

        status = segmentationMeta.getStatus()
        if status == self.STATUS.FLAGGED:
            return True

        return False

    def getLevel(self, versionTag="final") -> str:
        if self.isSegemented() is False or self.hasSegmentationMeta(tag=versionTag) is False:
            return ""

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=versionTag)
        if segmentationMeta is None:
            return ""
        return segmentationMeta.getLevel()

    def setSegmentationFileName(self, fileName: str):
        self.segmentationFileName = fileName

    def getSegmentationFileName(self) -> str:
        return self.segmentationFileName

    def setClientId(self, client_id: str):
        self.client_id = client_id

    def addNewSegmentationMeta(self, tag: str, status: str, level: str, approvedBy: str, comment: str):
        segmentationMeta = SegmentationMeta()
        segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment, editTime="")
        segmentationMeta.setVersionNumber(versionTag=tag)
        self.segmentationMetaDict[tag] = segmentationMeta

    def getSegmentationMetaByVersionTag(self, tag: str):
        if tag not in self.segmentationMetaDict:
            return None
        return self.segmentationMetaDict[tag]

    def isEqualSegmentationMeta(self, tag: str, status: str, level: str, approvedBy: str, comment: str) -> bool:
        segmentationMeta = self.getSegmentationMetaByVersionTag(tag)
        if (
            segmentationMeta is None
            and self.isBlank(status)
            and self.isBlank(level)
            and self.isBlank(approvedBy)
            and self.isBlank(comment)
        ):
            return True

        if segmentationMeta is None:
            self.addNewSegmentationMeta(tag, status, level, approvedBy, comment)
            return False

        return segmentationMeta.isEqual(status=status, level=level, approvedBy=approvedBy, comment=comment)

    def isBlank(self, string) -> bool:
        return not (string and string.strip())

    def getMetaByVersionTag(self, tag: str) -> dict:
        if tag not in self.segmentationMetaDict:
            return {}
        segmentationMeta = self.getSegmentationMetaByVersionTag(tag)
        return segmentationMeta.getMeta()

    def hasSegmentationMeta(self, tag="final") -> bool:

        segmentationMeta = self.getSegmentationMetaByVersionTag(tag=tag)
        if segmentationMeta is None:
            return False
        return True

    def addSegementationMetaByVersionTag(self, tag="", status="", level="", approvedBy="", comment=""):
        segmentationMeta = SegmentationMeta()
        segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment)
        segmentationMeta.setVersionNumber(versionTag=tag)
        self.segmentationMetaDict[tag] = segmentationMeta

    def getSegementationMetaByVersionTag(self, tag: str) -> SegmentationMeta:
        if self.isBlank(tag):
            return None
        if tag not in self.segmentationMetaDict.keys():
            return None
        return self.segmentationMetaDict[tag]

    def obtainUpdatedParams(self, tag: str) -> dict:
        params = self.labelContent.copy()
        segementationMeta = self.getSegementationMetaByVersionTag(tag=tag)
        if segementationMeta is None:
            return params
        segementationMeta.setEditTime()
        metaData = segementationMeta.getMeta()
        if len(metaData) > 0:
            params["segmentationMeta"] = metaData["segmentationMeta"]
        return params

    def updateSegmentationMetaByVerionTag(self, tag="", status="", level="", approvedBy="", comment="") -> bool:
        if self.isBlank(tag):
            return False
        segmentationMeta = self.getSegementationMetaByVersionTag(tag=tag)
        if segmentationMeta is None:
            segmentationMeta = SegmentationMeta()
            segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment)
            segmentationMeta.setVersionNumber(versionTag=tag)
        else:
            segmentationMeta.update(status=status, level=level, approvedBy=approvedBy, comment=comment)
        segmentationMeta.setEditTime()
        self.segmentationMetaDict[tag] = segmentationMeta

        return True

    def updateApprovedStatusOfOtherThanSubjectedVersion(
        self, subjectedTag: str, difficultyLevel: str
    ) -> Dict[str, dict]:
        tagToSegmentationMetaJson = {}
        for tag, segmentationMeta in self.segmentationMetaDict.items():
            if subjectedTag == tag:
                continue

            updated = False
            if segmentationMeta.getStatus() == self.STATUS.APPROVED:
                segmentationMeta.setStatus("")
                updated = True

            if segmentationMeta.getLevel != difficultyLevel:
                segmentationMeta.setLevel(difficultyLevel)
                updated = True

            if updated:
                self.segmentationMetaDict[tag] = segmentationMeta
                tagToSegmentationMetaJson[tag] = segmentationMeta.getMeta()
        return tagToSegmentationMetaJson

    def getApprovedVersionTagElseReturnLatestVersion(self) -> str:
        latest = 0
        latestVersion = ""

        if len(self.segmentationMetaDict) == 1:
            return [*self.segmentationMetaDict.keys()][0]

        for tag, segmentationMeta in self.segmentationMetaDict.items():

            if segmentationMeta.getStatus() == self.STATUS.APPROVED:
                return tag

            version = segmentationMeta.getVersionNumber()

            if latest < version:
                latest = version
                latestVersion = tag

        return latestVersion

    # methods dealing with versions

    def getLatestVersionTag(self) -> str:
        if len(self.versionNames) == 0:
            return ""
        return self.versionNames[len(self.versionNames) - 1]

    def getOldestVersion(self) -> str:
        if len(self.versionNames) == 0:
            return ""
        return self.versionNames[0]

    def getNewVersionName(self) -> str:
        subsequentIndex = self.obtainSubsequentIndexFromVersionName(self.versionNames)
        newVersionName = self.obtainNextVersionName(subsequentIndex)
        self.versionNames.append(newVersionName)
        return newVersionName

    def getNumberOfVersions(self) -> int:
        return len(self.versionNames)

    def getVersionName(self, version: int) -> str:
        if version >= len(self.versionNames):
            return ""

        return self.versionNames[version]

    def hasVersionTag(self, versionTag: str):
        return versionTag in self.versionNames

    def getVersionNames(self) -> List[str]:
        return self.versionNames

    def obtainNextVersionName(self, index: int) -> str:
        return self.prefixVersion + str(index)

    def deleteVersionName(self, versionTag: str):

        if versionTag not in self.versionNames:
            return

        self.versionNames.remove(versionTag)

        if versionTag in self.segmentationMetaDict.items():
            self.segmentationMetaDict.pop(versionTag)

    def obtainSubsequentIndexFromVersionName(self, versionNames) -> int:
        if len(versionNames) == 0:
            return 1
        lastVersionTag = versionNames[len(versionNames) - 1]
        if lastVersionTag == self.FINAL or lastVersionTag == self.ORIGIN:
            return 1
        try:
            indexOfDelimeter = lastVersionTag.index("_")
        except:
            exceptionIndex = len(versionNames) + 100
            logging.info(
                "Version name is incorrect. Format should be like 'version_1' but was {}. Hence, following id will be used {}.".format(
                    lastVersionTag, exceptionIndex
                )
            )
            return exceptionIndex

        lastCharIndex = len(lastVersionTag)
        versionTagIndex = lastVersionTag[indexOfDelimeter + 1 : lastCharIndex]
        return int(versionTagIndex) + 1

    def display(self):
        print("name: ", self.name)
        print("fileName: ", self.fileName)
        print("nodeName: ", self.nodeName)
        print("checksum: ", self.checkSum)
        print("isSegmented: ", self.segmented)
        print("getTimeStamp: ", self.getTimeOfAnnotation())
        print("=== Version labels ====")
        for version in self.versionNames:
            print("version: ", version)
        if self.isSegemented():
            print("Client Id: ", self.client_id)
            print("segmentationFileName: ", self.segmentationFileName)
            print("=== Segmentation Meta ====")

        if self.hasSegmentationMeta():
            for k, segmentationMeta in self.segmentationMetaDict.items():
                print("version: ", k)
                segmentationMeta.display()
