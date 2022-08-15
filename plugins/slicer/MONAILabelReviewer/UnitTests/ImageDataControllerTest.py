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

import json
import os
import sys
import unittest
from typing import Dict
from unittest.mock import Mock, patch

sys.path.append("..")
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.ImageDataController import ImageDataController
from MONAILabelReviewerLib.MONAILabelReviewerEnum import Level, SegStatus
from MONAILabelReviewerLib.MonaiServerREST import MonaiServerREST


class ImageDataControllerTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.STATUS = SegStatus()
        self.LEVEL = Level()
        self.url = "http://127.0.0.1:8000"
        self.controller = ImageDataController()

        self.nameToImageData: Dict[str, ImageData] = {}
        self.createImageData()
        self.testDataStore_V2_json = ""
        self.loadJsonStr()

    @classmethod
    def loadJsonStr(self) -> str:
        with open(os.path.join(sys.path[0], "TestDataSet/test_json_datastore_v2.json")) as f:
            self.testDataStore_V2_json = json.dumps(json.load(f))

    @classmethod
    def createImageData(self):

        # is segmented
        imageDataTest_1 = ImageData(
            name="6667571",
            fileName="6667571.dcm",
            nodeName="6667571.dcm",
            checkSum="SHA256:2a454e9ab8a33dc74996784163a362a53e04adcee2fd73a8b6299bf0ce5060d3",
            segmented=True,
            timeStamp=1639985938,
            comment="",
        )
        imageDataTest_1.setClientId("Test-Radiolgist-Segmented")
        imageDataTest_1.setSegmentationFileName("6667571.seg.nrrd")
        imageDataTest_1.addNewSegmentationMeta(
            tag="final", status=self.STATUS.APPROVED, level=self.LEVEL.HARD, approvedBy="Test-Reviewer", comment=""
        )

        # is segmented
        imageDataTest_2 = ImageData(
            name="imageId_2",
            fileName="fileName_2",
            nodeName="nodeName_2",
            checkSum="checkSum_2",
            segmented=True,
            timeStamp=1640171961,
            comment="comment_2",
        )
        imageDataTest_2.setClientId("client_id_1")
        imageDataTest_2.setSegmentationFileName("testSegementation_2.nrrd")
        imageDataTest_2.addNewSegmentationMeta(
            tag="final",
            status=self.STATUS.FLAGGED,
            level=self.LEVEL.MEDIUM,
            approvedBy="theRadologist_2",
            comment="comment_2",
        )

        # is not segmented
        imageDataTest_3 = ImageData(
            name="6213798",
            fileName="6213798.dcm",
            nodeName="6213798.dcm",
            checkSum="SHA256:5ca275af76a8fe88939058f9c91ecb72ce96bd5f012198fc366e4ef0214849b9",
            segmented=False,
            timeStamp=1642170835,
            comment="",
        )

        self.nameToImageData["6667571"] = imageDataTest_1
        self.nameToImageData["imageId_2"] = imageDataTest_2
        self.nameToImageData["6213798"] = imageDataTest_3

    @classmethod
    def areEqual(self, imageData_1: ImageData, imageData_2: ImageData) -> bool:
        if (
            imageData_1.getClientId() == imageData_2.getClientId()
            and imageData_1.getName() == imageData_2.getName()
            and imageData_1.getFileName() == imageData_2.getFileName()
            and imageData_1.getNodeName() == imageData_2.getNodeName()
            and imageData_1.isSegemented() == imageData_2.isSegemented()
        ):

            if imageData_1.isSegemented() is False:
                return True

            if (
                imageData_1.getSegmentationFileName() == imageData_2.getSegmentationFileName()
                and imageData_1.isApproved() == imageData_2.isApproved()
                and imageData_1.isFlagged() == imageData_2.isFlagged()
                and imageData_1.getLevel() == imageData_2.getLevel()
                and imageData_1.getApprovedBy() == imageData_2.getApprovedBy()
            ):
                return True

        return False

    @patch.object(MonaiServerREST, "getServerUrl", return_value="http://127.0.0.1:8000")
    def test_returnedUrl(self, getServerUrl):
        self.controller.setMonaiServer(self.url)
        url = self.controller.getServerUrl()
        getServerUrl.assert_called_once()
        self.assertEqual(self.url, url)

    @patch.object(MonaiServerREST, "checkServerConnection", return_value=True)
    def test_connectToMonaiServer(self, checkServerConnection):
        isConnected = self.controller.connectToMonaiServer(self.url)
        checkServerConnection.assert_called_once()
        self.assertTrue(isConnected)

    def test_getMapIdToImageData(self):
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)
        returnedMap = self.controller.getMapIdToImageData()

        selectedImageData = returnedMap["6213798"]
        expectedImageData = self.nameToImageData["6213798"]

        self.assertTrue(self.areEqual(expectedImageData, selectedImageData))

    def test_initMetaDataProcessing(self):
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)
        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

    def test_getStatistics(self):
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        statistics = self.controller.getStatistics()

        self.assertEqual(50, statistics["segmentationProgress"])
        self.assertEqual("2/4", statistics["idxTotalSegmented"])
        self.assertEqual("1/4", statistics["idxTotalApproved"])
        self.assertEqual(25, statistics["progressPercentage"])
        self.assertEqual(50, statistics["segmentationProgressAllPercentage"])
        self.assertEqual(25, statistics["approvalProgressPercentage"])

    def test_getClientIds(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        clientIds = self.controller.getClientIds()

        # Verify
        self.assertListEqual(clientIds, ["Test-Radiolgist-Segmented"])

    def test_getReviewers(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        clientIds = self.controller.getReviewers()

        # Verify
        self.assertListEqual(clientIds, ["Test-Reviewer"])

    def test_getAllImageData(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        imageDatas = self.controller.getAllImageData(
            segmented=True, isNotSegmented=False, isApproved=True, isFlagged=False
        )

        # Verify
        self.assertEqual(1, len(imageDatas))
        returnedImageData = imageDatas[0]
        expectedImageData = self.nameToImageData["6667571"]
        self.assertTrue(self.areEqual(expectedImageData, returnedImageData))

    def test_getImageDataByClientId(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        imageDatas = self.controller.getImageDataByClientId(
            selectedClientId="Test-Radiolgist-Segmented", isApproved=True, isFlagged=False
        )

        # Verify
        self.assertEqual(1, len(imageDatas))
        returnedImageData = imageDatas[0]
        expectedImageData = self.nameToImageData["6667571"]
        self.assertTrue(self.areEqual(expectedImageData, returnedImageData))

    def test_getPercentageApproved(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        percentageApprovedOfClient, idxApprovedOfClient = self.controller.getPercentageApproved(
            selectedClientId="Test-Radiolgist-Segmented"
        )

        # Verify
        self.assertEqual(50, percentageApprovedOfClient)
        self.assertEqual("1/2", idxApprovedOfClient)

    def test_getPercentageSemgmentedByClient(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        percentageSemgmentedByClient, idxSegmentedByClient = self.controller.getPercentageSemgmentedByClient(
            selectedClientId="Test-Radiolgist-Segmented"
        )

        # Verify
        self.assertEqual(50, percentageSemgmentedByClient)
        self.assertEqual("2/4", idxSegmentedByClient)

    def test_getMultImageDataByIds(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        imageIdDummy = "1234567"
        idToImage: Dict[str, ImageData] = self.controller.getMultImageDataByIds(
            imageIds=["6213798", "6667571", imageIdDummy]
        )

        # Verify
        self.assertTrue("6213798" in idToImage.keys())
        self.assertTrue("6667571" in idToImage.keys())
        self.assertTrue(imageIdDummy not in idToImage.keys())

        expectedImageData_1 = self.nameToImageData["6213798"]
        expectedImageData_2 = self.nameToImageData["6667571"]

        self.assertTrue(self.areEqual(expectedImageData_1, idToImage["6213798"]))
        self.assertTrue(self.areEqual(expectedImageData_2, idToImage["6667571"]))

    def test_searchByAnnotatorReviewer_selectedReviewer_is_all(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        idToImage: Dict[str, ImageData] = self.controller.searchByAnnotatorReviewer(
            selectedAnnotator="Test-Radiolgist-Segmented", selectedReviewer="All", isApproved=True, isFlagged=False
        )

        # Verify
        self.assertTrue("6667571" in idToImage.keys())
        expectedImageData = self.nameToImageData["6667571"]
        self.assertTrue(self.areEqual(expectedImageData, idToImage["6667571"]))

    def test_searchByAnnotatorReviewer_selectedReviewer_is_all_and_selectedReviewer_is_all(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        # Test
        idToImage: Dict[str, ImageData] = self.controller.searchByAnnotatorReviewer(
            selectedAnnotator="All", selectedReviewer="All", isApproved=True, isFlagged=False
        )

        # Verify
        self.assertTrue("6667571" in idToImage.keys())
        expectedImageData = self.nameToImageData["6667571"]
        self.assertTrue(self.areEqual(expectedImageData, idToImage["6667571"]))

    def test_updateLabelInfo_successfully_update_label_info(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        mockObject.updateLabelInfo = Mock(return_value=200)

        # Test
        successfullyUpdate = self.controller.updateLabelInfo(imageId="6667571", updatedMetaJson="")

        # Verify
        self.assertTrue(successfullyUpdate)

    def test_updateLabelInfo_failed_update_label_info(self):
        # Set up
        json_with_segmentation = json.loads(self.testDataStore_V2_json)

        self.controller.monaiServerREST = MonaiServerREST(self.url)
        mockObject = self.controller.monaiServerREST
        mockObject.requestDataStoreInfo = Mock(return_value=json_with_segmentation)

        success = self.controller.initMetaDataProcessing()
        self.assertTrue(success)

        mockObject.updateLabelInfo = Mock(return_value=400)

        # Test
        failedUpdate = self.controller.updateLabelInfo(imageId="6667571", updatedMetaJson="")

        # Verify
        self.assertFalse(failedUpdate)


if __name__ == "__main__":
    unittest.main()
