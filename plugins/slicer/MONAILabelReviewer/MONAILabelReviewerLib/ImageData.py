from datetime import datetime

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

        self.STATUS = SegStatus()

        self.client_id: str = None
        self.segmentationFileName: str = None
        self.tempDirectory: str = None
        self.segmentationMeta: SegmentationMeta = None

    def getName(self) -> str:
        return self.name

    def getFileName(self) -> str:
        return self.fileName

    def getNodeName(self) -> str:
        return self.nodeName

    def getCheckSum(self) -> str:
        return self.checkSum

    def getClientId(self) -> str:
        return self.client_id

    def getTimeStamp(self) -> int:
        return self.timeStamp

    def getTime(self) -> str:
        return str(datetime.fromtimestamp(self.timeStamp))

    def isSegemented(self) -> bool:
        return self.segmented

    def getComment(self) -> str:
        return self.comment

    def getStatus(self) -> str:
        if self.isSegemented() is False:
            return self.STATUS.NOT_SEGMENTED
        if self.hasSegmentationMeta() is False:
            return ""
        if self.hasSegmentationMeta():
            return self.segmentationMeta.getStatus()

    def getApprovedBy(self) -> str:
        if self.isSegemented() is False:
            return ""
        if self.hasSegmentationMeta() is False:
            return ""
        return self.segmentationMeta.getApprovedBy()

    def isApproved(self) -> bool:
        if self.hasSegmentationMeta() is False:
            return False
        if self.getStatus() == self.STATUS.APPROVED:
            return True
        return False

    def isFlagged(self) -> bool:
        if self.hasSegmentationMeta() is False:
            return False
        if self.getStatus() == self.STATUS.FLAGGED:
            return True
        return False

    def getLevel(self) -> str:
        if self.isSegemented() is False:
            return ""
        if self.hasSegmentationMeta() is False:
            return ""
        return self.segmentationMeta.getLevel()

    def setSegmentationFileName(self, fileName: str):
        self.segmentationFileName = fileName

    def getSegmentationFileName(self) -> str:
        return self.segmentationFileName

    def setClientId(self, client_id: str):
        self.client_id = client_id

    def setSegmentationMeta(self, status="", level="", approvedBy="", comment=""):
        self.segmentationMeta = SegmentationMeta()
        self.segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment)

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

        self.segmentationMeta.setStatus(status)
        self.segmentationMeta.setLevel(level)
        self.segmentationMeta.setApprovedBy(approvedBy)
        self.segmentationMeta.setComment(comment)

    def isBlank(self, string) -> bool:
        return not (string and string.strip())

    def getMeta(self) -> str:
        if self.segmentationMeta is None:
            return None
        return self.segmentationMeta.getMeta()

    def hasSegmentationMeta(self) -> bool:
        return self.segmentationMeta is not None

    def display(self):
        print("name: ", self.name)
        print("fileName: ", self.fileName)
        print("nodeName: ", self.nodeName)
        print("checksum: ", self.checkSum)
        print("isSegmented: ", self.segmented)
        print("getTimeStamp: ", self.getTime())
        if self.isSegemented():
            print("Client Id: ", self.client_id)
            print("segmentationFileName: ", self.segmentationFileName)
            print("=== Segmentation Meta ====")
        if self.hasSegmentationMeta():
            self.segmentationMeta.display()
