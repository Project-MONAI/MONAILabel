import json
import time
import logging

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

    def build(self, status="", level="", approvedBy="", comment="", editTime=""):
        self.setEditTime()
        self.status = status
        self.level = level
        self.approvedBy = approvedBy
        self.comment = comment
        self.editTime = editTime

    def update(self,  status="", level="", approvedBy="", comment="") -> bool:
        logging.warn("=============== HEER ==============")
        logging.warn("status={}, level={}, approvedBy={}, comment={}".format(status, level, approvedBy, comment))
        isChanged = False
        if(self.isBlank(status) is False and status != self.status):
            self.status = status
            isChanged = True

        if(self.isBlank(level) is False and level != self.level):
            self.level = level
            isChanged = True

        if(self.isBlank(comment) is False and comment != self.comment):
            self.comment = comment
            isChanged = True
        
        if(isChanged):
            if(self.isBlank(approvedBy) is False and approvedBy != self.approvedBy):
                self.approvedBy = approvedBy

            self.setEditTime()
            
        return isChanged

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
        self.editTime = int(time.time())

    def getMeta(self)-> dict:
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
        print("status: ", self.status)
        print("level: ", self.level)
        print("approvedBy: ", self.approvedBy)
        print("editTime: ", self.editTime)
        print("comment: ", self.comment)
