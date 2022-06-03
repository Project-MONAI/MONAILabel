from typing import Dict

from MONAILabelReviewerLib.DataStoreKeys import DataStoreKeys
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.MONAILabelReviewerEnum import Label

"""
JsonParser parses the datastore.json file
and caches the information in dictionary: Mapping from id to ImageData
"""


class JsonParser:
    def __init__(self, jsonObject: dict):
        self.LABEL = Label()
        self.SEGMENTATION_META = "segmentationMeta"

        self.dataStoreKeys = DataStoreKeys()
        self.jsonObject = jsonObject
        self.mapIdToImageData: Dict[str, ImageData] = {}

    def init(self):
        self.parseJsonToImageData()

    def getValueByKey(self, keyArr: list, jsonObj: dict):
        if len(keyArr) == 0:
            return ""
        for key in keyArr:
            if key not in jsonObj:
                return ""
            jsonObj = jsonObj[key]
        return jsonObj

    def getFileName(self, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.FILENAME, obj)

    def getNodeName(self, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.NODE_NAME, obj)

    def getCheckSum(self, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.CHECKSUM, obj)

    def getTimeStamp(self, obj: dict) -> int:
        if self.hasKeyAnnotate(obj):
            return self.getValueByKey(self.dataStoreKeys.TIMESTAMP_ANNOTATE, obj)
        if self.hasKeyRandom(obj):
            return self.getValueByKey(self.dataStoreKeys.TIMESTAMP_RANDOM, obj)
        return self.getValueByKey(self.dataStoreKeys.TIMESTAMP, obj)

    def getInfo(self, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.IMAGE_INFO, obj)

    def getInfoInLabels(self, label: str, obj: dict) -> Dict[str, str]:
        keys = self.dataStoreKeys.getInfoInLabels(label)
        return self.getValueByKey(keys, obj)

    def hasLabels(self, obj: dict) -> bool:
        lablesSection = self.getValueByKey(self.dataStoreKeys.LABELS, obj)
        if len(lablesSection) == 0:
            return False
        return True

    def hasSegemantatorsId(self, obj: dict) -> bool:
        labels = self.getValueByKey(self.dataStoreKeys.LABELS, obj)

        if len(labels) == 0:
            return False

        if self.dataStoreKeys.FINAL not in labels:
            return False

        final = self.getValueByKey(self.dataStoreKeys.LABELS_FINAL, obj)
        if self.dataStoreKeys.INFO not in final:
            return False

        info = self.getValueByKey(self.dataStoreKeys.LABELS_FINAL_INFO, obj)
        if self.dataStoreKeys.LABEL_INFO not in info:
            return False

        labelsInfo = self.getValueByKey(self.dataStoreKeys.LABELS_FINAL_INFO_LABELS_INFO, obj)
        if len(labelsInfo) == 0:
            return False
        return True

    def isSegmented(self, obj: dict) -> bool:
        labels = self.getValueByKey(self.dataStoreKeys.LABELS, obj)
        if len(labels) == 0:
            return False
        if self.dataStoreKeys.FINAL not in labels:
            return False
        return True

    def hasKeyFinal(self, obj: dict):
        labelsDict = self.getValueByKey(self.dataStoreKeys.LABELS, obj)
        return self.dataStoreKeys.FINAL in labelsDict

    def hasKeyOriginal(self, obj: dict):
        labelsDict = self.getValueByKey(self.dataStoreKeys.LABELS, obj)
        return self.dataStoreKeys.ORIGINAL in labelsDict

    def getSegmentationName(self, obj: dict) -> dict:
        if self.hasKeyFinal(obj):
            return self.getValueByKey(self.dataStoreKeys.SEGMENTATION_NAME_BY_FINAL, obj)
        if self.hasKeyOriginal(obj):
            return self.getValueByKey(self.dataStoreKeys.SEGMENTATION_NAME_BY_ORIGINAL, obj)

    def hasKeyAnnotate(self, obj: dict):
        strategyDict = self.getValueByKey(self.dataStoreKeys.STRATEGY, obj)
        return self.dataStoreKeys.ANNOTATE in strategyDict

    def hasKeyRandom(self, obj: dict):
        strategyDict = self.getValueByKey(self.dataStoreKeys.STRATEGY, obj)
        return self.dataStoreKeys.RANDOM in strategyDict

    def getClientId(self, obj: dict) -> str:
        if self.hasKeyAnnotate(obj):
            return self.getValueByKey(self.dataStoreKeys.CLIENT_ID_BY_ANNOTATE, obj)
        if self.hasKeyRandom(obj):
            return self.getValueByKey(self.dataStoreKeys.CLIENT_ID_BY_RANDOM, obj)
        if self.hasSegemantatorsId(obj):
            return self.getValueByKey(self.dataStoreKeys.CLIENT_ID, obj)
        return "Segmented without annotator's id"

    def getMetaStatus(self, label: str, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.getMetaStatus(label), obj)

    def getMetaLevel(self, label: str, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.getMetaLevel(label), obj)

    def getMetaApprovedBy(self, label: str, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.getMetaApprovedBy(label), obj)

    def getMetaEditTime(self, label: str, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.getMetaEditTime(label), obj)

    def getMetaComment(self, label: str, obj: dict) -> str:
        return self.getValueByKey(self.dataStoreKeys.getMetaComment(label), obj)

    def parseJsonToImageData(self):
        objects = self.jsonObject[self.dataStoreKeys.OBJECT]
        counter = 0
        for key, value in objects.items():
            imageData = self.jsonToImageData(key, value)
            self.mapIdToImageData[key] = imageData
            counter += 1

    def jsonToImageData(self, key, value):
        fileName = self.getFileName(value)
        nodeName = self.getNodeName(value)
        checksum = self.getCheckSum(value)
        isSegmented = self.isSegmented(value)
        timeStamp = self.getTimeStamp(value)
        imageData = ImageData(
            name=key,
            fileName=fileName,
            nodeName=nodeName,
            checkSum=checksum,
            segmented=isSegmented,
            timeStamp=timeStamp,
        )

        if isSegmented:
            segName = self.getSegmentationName(value)
            imageData.setSegmentationFileName(segName)

            clientId = self.getClientId(value)
            imageData.setClientId(clientId)

        if self.hasLabels(value) is False:
            return imageData

        label = ""
        if self.hasKeyFinal(value):
            label = self.LABEL.FINAL

        if self.hasKeyOriginal(value):
            label = self.LABEL.ORGINAL

        info = self.getInfoInLabels(label=label, obj=value)
        if self.hasSegmentationMeta(info):
            status = self.getMetaStatus(label, value)
            level = self.getMetaLevel(label, value)
            approvedBy = self.getMetaApprovedBy(label, value)
            comment = self.getMetaComment(label, value)
            imageData.setSegmentationMeta(status, level, approvedBy, comment)
        return imageData

    def hasSegmentationMeta(self, info: dict) -> bool:
        return self.SEGMENTATION_META in info.keys()

    def getMapIdToImageData(self):
        return self.mapIdToImageData
