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

import sys
import unittest
from typing import Dict, List

sys.path.append("..")
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.ImageDataExtractor import ImageDataExtractor
from MONAILabelReviewerLib.MONAILabelReviewerEnum import Level, SegStatus


class ImageDataExtractorTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.STATUS = SegStatus()
        self.LEVEL = Level()
        self.createImageData()

    @classmethod
    def createImageData(self):

        # is segmented
        imageDataTest_1 = ImageData(
            name="imageId_1",
            fileName="fileName_1",
            nodeName="nodeName_1",
            checkSum="checkSum_1",
            segmented=True,
            timeStamp=1640171961,
            comment="comment_1",
        )
        imageDataTest_1.setClientId("client_id_1")
        imageDataTest_1.setSegmentationFileName("testSegementation_1.nrrd")
        imageDataTest_1.addNewSegmentationMeta(
            tag="final",
            status=self.STATUS.APPROVED,
            level=self.LEVEL.HARD,
            approvedBy="theRadologist_1",
            comment="comment_1",
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
            name="imageId_3",
            fileName="fileName_3",
            nodeName="nodeName_3",
            checkSum="checkSum_3",
            segmented=False,
            timeStamp=1640171961,
            comment="comment_3",
        )
        imageDataTest_3.setClientId("client_id_3")

        self.nameToImageData: Dict[str, ImageData] = {}
        self.nameToImageData["imageId_1"] = imageDataTest_1
        self.nameToImageData["imageId_2"] = imageDataTest_2
        self.nameToImageData["imageId_3"] = imageDataTest_3
        self.imageDataExtractor = ImageDataExtractor(nameToImageData=self.nameToImageData)
        self.imageDataExtractor.init()

    @classmethod
    def areEqual(self, imageData_1: ImageData, imageData_2: ImageData) -> bool:
        if (
            imageData_1.getClientId() == imageData_2.getClientId()
            and imageData_1.getName() == imageData_2.getName()
            and imageData_1.getFileName() == imageData_2.getFileName()
            and imageData_1.getNodeName() == imageData_2.getNodeName()
            and imageData_1.isSegemented() == imageData_2.isSegemented()
            and imageData_1.getSegmentationFileName() == imageData_2.getSegmentationFileName()
            and imageData_1.isApproved() == imageData_2.isApproved()
            and imageData_1.isFlagged() == imageData_2.isFlagged()
            and imageData_1.getLevel() == imageData_2.getLevel()
            and imageData_1.getApprovedBy() == imageData_2.getApprovedBy()
        ):
            return True
        return False

    def test_getTotalNumImages(self):
        totalNumOfImages = self.imageDataExtractor.getTotalNumImages()
        print(totalNumOfImages)
        self.assertEqual(len(self.nameToImageData), totalNumOfImages)

    def test_getImageDataIds(self):
        ids = self.imageDataExtractor.getImageDataIds()
        expectedIds = [*self.nameToImageData.keys()]
        containsAll = all(id in ids for id in expectedIds)
        self.assertEqual(True, containsAll)

    def test_getClientIds(self):

        expectedClientIds = ["client_id_1"]
        clientIds = self.imageDataExtractor.getClientIds()
        self.assertEqual(len(expectedClientIds), len(clientIds))
        containsClients = all(id in clientIds for id in expectedClientIds)
        self.assertEqual(True, containsClients)

    def test_getReviewers(self):
        expectedReviewersIds = ["theRadologist_1", "theRadologist_2"]
        reviewerIds = self.imageDataExtractor.getReviewers()
        self.assertEqual(len(expectedReviewersIds), len(reviewerIds))
        containsReviewers = all(id in reviewerIds for id in expectedReviewersIds)
        self.assertEqual(True, containsReviewers)

    def test_getImageDataNotsegmented(self):
        notSegmentedImages: List[ImageData] = self.imageDataExtractor.getImageDataNotsegmented()
        self.assertEqual(1, len(notSegmentedImages))
        notSegementedImage = notSegmentedImages[0]
        self.assertEqual("client_id_3", notSegementedImage.getClientId())
        self.assertEqual(False, notSegementedImage.isSegemented())

    def test_getNumOfNotSegmented(self):
        numOfNotSegmented = self.imageDataExtractor.getNumOfNotSegmented()
        self.assertEqual(1, numOfNotSegmented)

    def test_getNumOfSegmented(self):
        numOfNotSegmented = self.imageDataExtractor.getNumOfSegmented()
        self.assertEqual(2, numOfNotSegmented)

    def test_getSegmentationProgessInPercentage(self):
        percentage = self.imageDataExtractor.getSegmentationProgessInPercentage()
        self.assertEqual(66, percentage)

    def test_getSegmentationProgessInPercentage_as_fraction(self):
        idxTotalSegmented = self.imageDataExtractor.getSegmentationVsTotalStr()
        self.assertEqual("2/3", idxTotalSegmented)

    def test_getApprovalProgressInPercentage(self):
        fraction = self.imageDataExtractor.getApprovalProgressInPercentage()
        self.assertEqual(33, fraction)

    def test_getApprovalVsTotal(self):
        idxTotalApproved = self.imageDataExtractor.getApprovalVsTotal()
        self.assertEqual("1/3", idxTotalApproved)

    def test_getAllImageData_segmented_is_true_approved_is_true(self):
        imageDatas = self.imageDataExtractor.getAllImageData(
            segmented=True, notSegmented=False, approved=True, flagged=False
        )
        self.assertEqual(1, len(imageDatas))
        expectedImageData = self.nameToImageData["imageId_1"]
        self.assertEqual(True, self.areEqual(expectedImageData, imageDatas[0]))

    def test_getAllImageData_segmented_is_true_flagges_is_true(self):
        imageDatas = self.imageDataExtractor.getAllImageData(
            segmented=True, notSegmented=False, approved=False, flagged=True
        )
        self.assertEqual(1, len(imageDatas))
        expectedImageData = self.nameToImageData["imageId_2"]
        self.assertEqual(True, self.areEqual(expectedImageData, imageDatas[0]))

    def test_getAllImageData_isNotSegmented_is_false_approved_is_false(self):
        imageDatas = self.imageDataExtractor.getAllImageData(
            segmented=False, notSegmented=True, approved=False, flagged=False
        )
        self.assertEqual(1, len(imageDatas))
        expectedImageData = self.nameToImageData["imageId_3"]
        self.assertEqual(True, self.areEqual(expectedImageData, imageDatas[0]))

    def test_getImageDataByClientId_approved_is_true(self):

        imageDataTest_4 = ImageData(
            name="imageId_4",
            fileName="fileName_4",
            nodeName="nodeName_4",
            checkSum="checkSum_4",
            segmented=True,
            timeStamp=1640171961,
            comment="comment_4",
        )
        imageDataTest_4.setClientId("client_id_1")
        imageDataTest_4.setSegmentationFileName("testSegementation_4.nrrd")
        imageDataTest_4.addNewSegmentationMeta(
            tag="final",
            status=self.STATUS.APPROVED,
            level=self.LEVEL.MEDIUM,
            approvedBy="theRadologist_4",
            comment="comment_4",
        )
        self.nameToImageData["imageId_4"] = imageDataTest_4

        imageDataExtractor = ImageDataExtractor(nameToImageData=self.nameToImageData)
        imageDataExtractor.init()
        returnedImageDatas = imageDataExtractor.getImageDataByClientId(
            clientId="client_id_1", approved=True, flagged=False
        )
        self.assertEqual(2, len(returnedImageDatas))
        returnedImageData_1 = list(filter(lambda image: (image.getName() == "imageId_1"), returnedImageDatas))
        returnedImageData_4 = list(filter(lambda image: (image.getName() == "imageId_4"), returnedImageDatas))

        self.assertEqual(True, self.areEqual(self.nameToImageData["imageId_1"], returnedImageData_1[0]))
        self.assertEqual(True, self.areEqual(self.nameToImageData["imageId_4"], returnedImageData_4[0]))

    def test_getImageDataByClientAndReviewer_approved_is_true(self):
        imageDataExtractor = ImageDataExtractor(nameToImageData=self.nameToImageData)
        imageDataExtractor.init()
        returnedImageDatas = imageDataExtractor.getImageDataByClientAndReviewer(
            clientId="client_id_1", reviewerId="theRadologist_1", approved=True, flagged=False
        )
        for imageData in returnedImageDatas:
            imageData.display()

        self.assertEqual(1, len(returnedImageDatas))
        imageData = returnedImageDatas[0]
        self.assertEqual(True, self.areEqual(self.nameToImageData["imageId_1"], imageData))

    def test_getImageDataByClientId_flagged_is_true(self):
        returnedImageDatas = self.imageDataExtractor.getImageDataByClientId(
            clientId="client_id_1", approved=False, flagged=True
        )
        self.assertEqual(1, len(returnedImageDatas))
        returnedImageData = returnedImageDatas[0]
        expectedImageData = self.nameToImageData["imageId_2"]
        self.assertEqual(True, self.areEqual(expectedImageData, returnedImageData))

    def test_getImageDataByReviewer_approved_is_true(self):
        returnedImageDatas = self.imageDataExtractor.getImageDataByReviewer(
            reviewerId="theRadologist_1", approved=True, flagged=False
        )
        self.assertEqual(1, len(returnedImageDatas))
        returnedImageData = returnedImageDatas[0]
        expectedImageData = self.nameToImageData["imageId_1"]
        self.assertEqual(True, self.areEqual(expectedImageData, returnedImageData))

    def test_getImageDataByReviewer_flagged_is_true(self):
        returnedImageDatas = self.imageDataExtractor.getImageDataByReviewer(
            reviewerId="theRadologist_2", approved=False, flagged=True
        )
        self.assertEqual(1, len(returnedImageDatas))
        returnedImageData = returnedImageDatas[0]
        expectedImageData = self.nameToImageData["imageId_2"]
        self.assertEqual(True, self.areEqual(expectedImageData, returnedImageData))

    def test_getImageDataByLevel_level_hard(self):
        returnedImageDatas = self.imageDataExtractor.getImageDataByLevel(isEasy=False, isMedium=False, isHard=True)
        self.assertEqual(1, len(returnedImageDatas))
        returnedImageData = returnedImageDatas["imageId_1"]
        expectedImageData = self.nameToImageData["imageId_1"]
        self.assertEqual(True, self.areEqual(expectedImageData, returnedImageData))

    def test_getSingleImageDataById_found_ImageData(self):
        returnedImageData = self.imageDataExtractor.getSingleImageDataById(imageId="imageId_1")
        expectedImageData = self.nameToImageData["imageId_1"]
        self.assertTrue(self.areEqual(expectedImageData, returnedImageData))

    def test_getSingleImageDataById_isBlank(self):
        returnedImageData = self.imageDataExtractor.getSingleImageDataById(imageId=" ")
        self.assertIsNone(returnedImageData)

    def test_getMultImageDataByIds(self):
        returnedImageDatas: Dict[str, ImageData] = self.imageDataExtractor.getMultImageDataByIds(
            ids=["imageId_1", "imageId_3", "dummy"]
        )
        self.assertEqual(2, len(returnedImageDatas))

        imageDataWithimageId_1 = returnedImageDatas["imageId_1"]
        imageDataWithimageId_3 = returnedImageDatas["imageId_3"]
        self.assertTrue(self.areEqual(self.nameToImageData["imageId_1"], imageDataWithimageId_1))
        self.assertTrue(self.areEqual(self.nameToImageData["imageId_3"], imageDataWithimageId_3))

    def test_getMultImageDataByIds_given_empty_idList(self):
        returnedImageDatas: Dict[str, ImageData] = self.imageDataExtractor.getMultImageDataByIds(ids=[])
        self.assertEqual(0, len(returnedImageDatas))

    def test_getNumApprovedSegmentation(self):
        numApprovedSegmentation = self.imageDataExtractor.getNumApprovedSegmentation()
        self.assertEqual(1, numApprovedSegmentation)

    def test_getPercentageApproved(self):
        imageDataTest_4 = ImageData(
            name="imageId_4",
            fileName="fileName_4",
            nodeName="nodeName_4",
            checkSum="checkSum_4",
            segmented=True,
            timeStamp=1640171961,
            comment="comment_4",
        )
        imageDataTest_4.setClientId("client_id_1")
        imageDataTest_4.setSegmentationFileName("testSegementation_4.nrrd")
        imageDataTest_4.addNewSegmentationMeta(
            tag="final",
            status=self.STATUS.APPROVED,
            level=self.LEVEL.MEDIUM,
            approvedBy="theRadologist_4",
            comment="comment_4",
        )
        self.nameToImageData["imageId_4"] = imageDataTest_4

        imageDataExtractor = ImageDataExtractor(nameToImageData=self.nameToImageData)
        imageDataExtractor.init()

        precentage, idxApprovedOfClient = imageDataExtractor.getPercentageApproved(clientId="client_id_1")
        self.assertEqual(66, precentage)
        self.assertEqual("2/3", idxApprovedOfClient)

    def test_getPercentageSemgmentedByClient(self):
        imageDataTest_4 = ImageData(
            name="imageId_4",
            fileName="fileName_4",
            nodeName="nodeName_4",
            checkSum="checkSum_4",
            segmented=True,
            timeStamp=1640171961,
            comment="comment_4",
        )
        imageDataTest_4.setClientId("client_id_1")
        imageDataTest_4.setSegmentationFileName("testSegementation_4.nrrd")
        imageDataTest_4.addNewSegmentationMeta(
            tag="final",
            status=self.STATUS.APPROVED,
            level=self.LEVEL.MEDIUM,
            approvedBy="theRadologist_4",
            comment="comment_4",
        )
        self.nameToImageData["imageId_4"] = imageDataTest_4

        imageDataExtractor = ImageDataExtractor(nameToImageData=self.nameToImageData)
        imageDataExtractor.init()

        precentage, idxSegmentedByClien = imageDataExtractor.getPercentageSemgmentedByClient(clientId="client_id_1")
        self.assertEqual(75, precentage)
        self.assertEqual("3/4", idxSegmentedByClien)

    def test_getApprovedSegmentationIds(self):
        idsOfApprovedSementations = self.imageDataExtractor.getApprovedSegmentationIds()
        self.assertEqual(1, len(idsOfApprovedSementations))

    def test_getSegmentedImageIds(self):
        idsOfSegmented = self.imageDataExtractor.getSegmentedImageIds()
        self.assertEqual(2, len(idsOfSegmented))


if __name__ == "__main__":
    unittest.main()
