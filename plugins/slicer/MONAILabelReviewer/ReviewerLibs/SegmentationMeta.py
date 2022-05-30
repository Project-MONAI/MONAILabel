import json
import time

"""
SegmentationMeta stores all the meta data of its corresponding ImageData
The class returns a json string which will be send to MONAI-Server to persist the
information in datastore.json
"""


class SegmentationMeta:
    def __init__(self):
        self.preFix = "params="
        self.status: str = ""
        self.level: str = ""
        self.approvedBy: str = ""
        self.editTime: str = ""
        self.comment: str = ""

    def build(self, status="", level="", approvedBy="", comment=""):
        self.setEditTime()
        self.status = status
        self.level = level
        self.approvedBy = approvedBy
        self.comment = comment

    def setComment(self, comment: str):
        self.setEditTime()
        self.comment = comment

    def setApprovedBy(self, approvedBy: str):
        self.setEditTime()
        self.approvedBy = approvedBy

    def setStatus(self, status: str):
        self.setEditTime()
        self.status = status

    def setLevel(self, level: str):
        self.setEditTime()
        self.level = level

    def setComment(self, comment: str):
        self.setEditTime()
        self.comment = comment

    def setEditTime(self):
        self.editTime = str(time.ctime())

    def getMeta(self) -> str:
        self.setEditTime()
        metaJson = {
            "segmentationMeta": {
                "status": self.status,
                "approvedBy": self.approvedBy,
                "level": self.level,
                "comment": self.comment,
                "editTime": self.editTime,
            }
        }
        jsonStr = json.dumps(metaJson)
        return self.preFix + jsonStr

    def getStatus(self) -> str:
        return self.status

    def getLevel(self) -> str:
        return self.level

    def getApprovedBy(self) -> str:
        return self.approvedBy

    def getComment(self) -> str:
        return self.comment

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

    def display(self):
        print("status: ", self.status)
        print("level: ", self.level)
        print("approvedBy: ", self.approvedBy)
        print("editTime: ", self.editTime)
        print("comment: ", self.comment)
