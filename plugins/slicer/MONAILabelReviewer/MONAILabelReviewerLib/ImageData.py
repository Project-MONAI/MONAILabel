from datetime import datetime
import logging
from typing import List, Dict

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
        self.name: str = name
        self.fileName: str = fileName
        self.nodeName: str = nodeName
        self.checkSum: str = checkSum
        self.segmented: bool = segmented
        self.timeStamp: int = timeStamp
        self.comment: str = comment

        self.versionNames : List[str] = [] # equals to labelNames
        self.labelContent : dict = {}
        self.segmentationMetaDict : Dict[str, SegmentationMeta] = {}

        self.STATUS = SegStatus()

        self.client_id: str = None
        self.segmentationFileName: str = None
        self.tempDirectory: str = None
        self.segmentationMeta: SegmentationMeta = None
        self.prefixVersion = "version_"
        self.FINAL = "final"
        self.ORIGIN = "origin"

    def setVersionNames(self, versionNames : List[str]):
        self.versionNames : List[str] = versionNames

    def setLabelContent(self, labelContent : dict):
        self.labelContent : dict = labelContent

    def setSegmentationMetaDict(self,segmentationMetaDict : Dict[str, SegmentationMeta]):
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

    def getClientId(self, versionTag = "final") -> str:
        if(versionTag == self.FINAL or versionTag == self.ORIGIN):
            return self.client_id
        if(versionTag in self.segmentationMetaDict.keys()):
            return self.segmentationMetaDict[versionTag].getApprovedBy()

    def getTimeStamp(self) -> int:
        return self.timeStamp

    def formatTimeStamp(self, timeStamp) -> str:
        if(type(timeStamp) == str):
            return timeStamp
        return str(datetime.fromtimestamp(timeStamp))

    def getTimeOfAnnotation(self) -> str:
        return self.formatTimeStamp(self.timeStamp)

    def getTimeOfEditing(self, versionTag = "final"):
        if(self.hasSegmentationMeta() is True 
                and (versionTag == self.FINAL or versionTag == self.ORIGIN)):
                formattedTime = self.formatTimeStamp(self.segmentationMeta.getEditTime())
                return formattedTime

        if(versionTag in self.segmentationMetaDict.keys()):
            formattedTime = self.formatTimeStamp(self.segmentationMetaDict[versionTag].getEditTime())
            return formattedTime

    def isSegemented(self) -> bool:
        return self.segmented

    def getLabelContent(self) -> dict:
        return self.labelContent

    def getComment(self, versionTag = "final") -> str:
        if(self.hasSegmentationMeta() is True 
                and (versionTag == self.FINAL or versionTag == self.ORIGIN)):
            return self.segmentationMeta.getComment()

        if(versionTag in self.segmentationMetaDict.keys()):
            return self.segmentationMetaDict[versionTag].getComment()

    def getSegmentationMetaDict(self) -> dict:
        return self.segmentationMetaDict

    def getStatus(self, versionTag = "final") -> str:
        if self.isSegemented() is False:
            return self.STATUS.NOT_SEGMENTED
        if(self.hasSegmentationMeta() is True 
                and (versionTag == self.FINAL or versionTag == self.ORIGIN)):
            return self.segmentationMeta.getStatus()
        
        if(versionTag in self.segmentationMetaDict.keys()):
            return self.segmentationMetaDict[versionTag].getStatus()

    def getApprovedBy(self, versionTag = "final") -> str:
        if self.isSegemented() is False:
            return ""

        if(self.hasSegmentationMeta() is True 
                and (versionTag == self.FINAL or versionTag == self.ORIGIN)):
            return self.segmentationMeta.getApprovedBy()

        if(versionTag in self.segmentationMetaDict.keys()):
            return self.segmentationMetaDict[versionTag].getApprovedBy()


    def isApproved(self, versionTag = "final") -> bool:
        if (self.hasSegmentationMeta() is True 
                and (versionTag == self.FINAL or versionTag == self.ORIGIN)
                and self.getStatus() == self.STATUS.APPROVED):
            return True

        if(versionTag in self.segmentationMetaDict.keys()):
            status = self.segmentationMetaDict[versionTag].getStatus()
            if (status == self.STATUS.APPROVED):
                return True
        return False

    def isFlagged(self, versionTag = "final") -> bool:
        if (self.hasSegmentationMeta() is True 
                and (versionTag == self.FINAL or versionTag == self.ORIGIN)
                and self.getStatus() == self.STATUS.FLAGGED):
            return True
        
        if(versionTag in self.segmentationMetaDict.keys()):
            meta = self.segmentationMetaDict[versionTag]
            print(meta)
            status = self.segmentationMetaDict[versionTag].getStatus()
            if (status == self.STATUS.FLAGGED):
                return True

        return False

    def getLevel(self, versionTag = "final") -> str:
        if(self.hasSegmentationMeta() is True 
                and (versionTag == self.FINAL or versionTag == self.ORIGIN)):
            return self.segmentationMeta.getLevel()

        if(versionTag in self.segmentationMetaDict.keys()):
            return self.segmentationMetaDict[versionTag].getLevel()


    def setSegmentationFileName(self, fileName: str):
        self.segmentationFileName = fileName

    def getSegmentationFileName(self) -> str:
        return self.segmentationFileName

    def setClientId(self, client_id: str):
        self.client_id = client_id

    def setSegmentationMeta(self, status="", level="", approvedBy="", comment="", editTime=""):
        self.segmentationMeta = SegmentationMeta()
        self.segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment, editTime=editTime)

    def isEqualSegmentationMeta(self, status="", level="", approvedBy="", comment="") -> bool:
        if (
            self.segmentationMeta is None
            and self.isBlank(status)
            and self.isBlank(level)
            and self.isBlank(approvedBy)
            and self.isBlank(comment)
        ):
            return True

        if self.segmentationMeta is None:
            self.setSegmentationMeta(status, level, approvedBy, comment)
            return False

        return self.segmentationMeta.isEqual(status=status, level=level, approvedBy=approvedBy, comment=comment)

    def updateSegmentationMeta(self, status="", level="", approvedBy="", comment=""):
        if self.segmentationMeta is None:
            self.segmentationMeta = SegmentationMeta()
            self.segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment)
            return
        self.segmentationMeta.update( status=status, level=level, approvedBy=approvedBy, comment=comment)

    def isBlank(self, string) -> bool:
        return not (string and string.strip())

    def getMeta(self) -> dict:
        if self.segmentationMeta is None:
            return ""
        return self.segmentationMeta.getMeta()

    def hasSegmentationMeta(self) -> bool:
        return self.segmentationMeta is not None

    def addSegementationMetaByVersionTag(self, tag="",  status="", level="", approvedBy="", comment=""):
        segmentationMeta = SegmentationMeta()
        segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment)
        self.segmentationMetaDict[tag] = segmentationMeta

    def getSegementationMetaByVersionTag(self, tag : str) -> SegmentationMeta:
        if (self.isBlank(tag)):
            return None
        if (tag not in self.segmentationMetaDict.keys()):
            return None
        return self.segmentationMetaDict[tag]

    def updateSegmentationMetaByVerionTag(self, tag="", status="", level="", approvedBy="", comment="") -> bool:
        if (self.isBlank(tag)):
            return False
        segmentationMeta = self.getSegementationMetaByVersionTag(tag = tag)
        if(segmentationMeta is None):
            segmentationMeta = SegmentationMeta()
            segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment)
        else:
            segmentationMeta.update(status=status, level=level, approvedBy=approvedBy, comment=comment)
        segmentationMeta.setEditTime()
        self.segmentationMetaDict[tag] = segmentationMeta
        return True


    #methods dealing with versions

    def getLatestVersionTag(self) -> str:
        if(len(self.versionNames) == 0):
            return ""
        return self.versionNames[len(self.versionNames) - 1]

    def getOldestVersion(self) -> str:
        if(len(self.versionNames) == 0):
            return ""
        return self.versionNames[0]

    def getNewVersionName(self) -> str:
        subsequentIndex = self.obtainSubsequentIndexFromVersionName(self.versionNames)
        newVersionName = self.obtainNextVersionName(subsequentIndex)
        self.versionNames.append(newVersionName)
        return newVersionName

    def getNumberOfVersions(self) -> int:
        return len(self.versionNames)

    def getVersionName(self, version : int) -> str:
        if (version >= len(self.versionNames)):
            return ""
        
        return self.versionNames[version]

    def hasVersionTag(self, versionTag : str ):
        return (versionTag in self.versionNames)

    def getVersionNames(self) -> List[str]:
        return self.versionNames
    
    def obtainNextVersionName(self, index : int) -> str:
        return self.prefixVersion + str(index)

    def deleteVersionName(self, versionTag : str):

        if (versionTag not in self.versionNames):
            return

        self.versionNames.remove(versionTag)

        if(versionTag in self.segmentationMetaDict.items()):
            self.segmentationMetaDict.pop(versionTag)

    def obtainSubsequentIndexFromVersionName(self, versionNames) -> int:
        if(len(versionNames) == 0):
            return 1
        lastVersionTag = versionNames[len(versionNames)-1]
        if(lastVersionTag == self.FINAL or lastVersionTag == self.ORIGIN):
            return 1
        try:
            indexOfDelimeter = lastVersionTag.index('_')
        except:
            exceptionIndex = len(versionNames) + 100
            logging.info("Version name is incorrect. Format should be like 'version_1' but was {}. Hence, following id will be used {}.".format(lastVersionTag, exceptionIndex))
            return exceptionIndex
            
        lastCharIndex = len(lastVersionTag)
        versionTagIndex = lastVersionTag[indexOfDelimeter+1:lastCharIndex]
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
            self.segmentationMeta.display()
            for k, segmentationMeta in self.segmentationMetaDict.items():
                print("version: ", k)
                segmentationMeta.display()
