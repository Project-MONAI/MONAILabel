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
import time

from MONAILabelReviewerLib.MONAILabelReviewerEnum import Label

"""
SegmentationMeta stores all the meta data of its corresponding ImageData
The class returns a json string which will be send to MONAI-Server to persist the
information in datastore.json
"""


class SegmentationMeta:
    def __init__(self):
        self.preFix = "params="
        self.LABEL = Label()

        self.status: str = ""
        self.level: str = ""
        self.approvedBy: str = ""
        self.editTime: str = ""
        self.comment: str = ""

        self.versionNumber: int = 0

    def build(self, status="", level="", approvedBy="", comment="", editTime=""):
        self.setEditTime()
        self.status = status
        self.level = level
        self.approvedBy = approvedBy
        self.comment = comment
        self.editTime = editTime

    def setVersionNumber(self, versionTag: str):
        if versionTag == self.LABEL.FINAL or versionTag == self.LABEL.ORIGINAL:
            self.versionNumber = 0
        else:
            self.versionNumber = self.parsNumberFromVersionTagString(versionTag=versionTag)

    def parsNumberFromVersionTagString(self, versionTag: str) -> int:
        lastCharIndex = len(versionTag)
        indexOfDelimeter = versionTag.index("_")
        versionTagIndex = versionTag[indexOfDelimeter + 1 : lastCharIndex]
        return int(versionTagIndex)

    def getVersionNumber(self) -> int:
        return self.versionNumber

    def update(self, status="", level="", approvedBy="", comment="") -> bool:
        logging.warn("=============== HEER ==============")
        logging.warn(f"status={status}, level={level}, approvedBy={approvedBy}, comment={comment}")
        isChanged = False
        if self.isBlank(status) is False and status != self.status:
            self.status = status
            isChanged = True

        if self.isBlank(level) is False and level != self.level:
            self.level = level
            isChanged = True

        if self.isBlank(comment) is False and comment != self.comment:
            self.comment = comment
            isChanged = True

        if isChanged:
            if self.isBlank(approvedBy) is False and approvedBy != self.approvedBy:
                self.approvedBy = approvedBy

        return isChanged

    def setApprovedBy(self, approvedBy: str):
        self.approvedBy = approvedBy

    def setStatus(self, status: str):
        self.status = status

    def setLevel(self, level: str):
        self.level = level

    def setComment(self, comment: str):
        self.comment = comment

    def setEditTime(self):
        self.editTime = int(time.time())

    def getMeta(self) -> dict:
        metaJson = {
            "segmentationMeta": {
                "status": self.status,
                "approvedBy": self.approvedBy,
                "level": self.level,
                "comment": self.comment,
                "editTime": self.editTime,
            }
        }
        return metaJson

    def getStatus(self) -> str:
        return self.status

    def getLevel(self) -> str:
        return self.level

    def getApprovedBy(self) -> str:
        return self.approvedBy

    def getComment(self) -> str:
        return self.comment

    def getEditTime(self) -> str:
        return self.editTime

    def isEqual(self, status="", level="", approvedBy="", comment=""):
        if status != self.status:
            return False
        if approvedBy != self.approvedBy:
            return False
        if level != self.level:
            return False
        if comment != self.comment:
            return False
        return True

    def isBlank(self, string) -> bool:
        return not (string and string.strip())

    def display(self):
        print("versionNumber: ", self.getVersionNumber)
        print("status: ", self.status)
        print("level: ", self.level)
        print("approvedBy: ", self.approvedBy)
        print("editTime: ", self.editTime)
        print("comment: ", self.comment)
