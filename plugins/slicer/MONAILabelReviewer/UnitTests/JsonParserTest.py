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

sys.path.append("..")
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.JsonParser import JsonParser


class JsonParserTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.jsonParser = JsonParser(None)

        fileNameWithOutSegmentation = "test_datastore_v2_image_without_segmentation.json"
        json_without_segmentation_str = self.loadJsonStr(fileNameWithOutSegmentation)

        self.json_without_segmentation = json.loads(json_without_segmentation_str)

        fileNameWithSegmentation = "test_datastore_v2_image_with_segmentation.json"
        json_with_segmentation_str = self.loadJsonStr(fileNameWithSegmentation)
        self.json_with_segmentation = json.loads(json_with_segmentation_str)

        fileNameWithMultipleVersions = "test_datastore_v2_image_with_multiple_versions.json"
        fileNameWithMultipleVersions_str = self.loadJsonStr(fileNameWithMultipleVersions)
        self.json_with_multiple_versions = json.loads(fileNameWithMultipleVersions_str)

    @classmethod
    def loadJsonStr(self, fileName: str) -> str:
        with open(os.path.join(sys.path[0], "TestDataSet/" + fileName)) as f:
            data = json.dumps(json.load(f))
        return data

    def test_getFileName(self):
        name = self.jsonParser.getFileName(self.json_without_segmentation)
        self.assertEqual(name, "6245968.dcm")

    def test_getNodeName(self):
        name = self.jsonParser.getNodeName(self.json_without_segmentation)
        self.assertEqual(name, "6245968.dcm")

    def test_getCheckSum(self):
        name = self.jsonParser.getCheckSum(self.json_without_segmentation)
        self.assertEqual(name, "SHA256:f1f8ef13433b1f0966e589818f3180750606eff69b3bc4a55e0181d7a9da8da1")

    def test_getTimeStamp(self):
        name = self.jsonParser.getTimeStamp(self.json_without_segmentation)
        self.assertEqual(name, 1642371057)

    def test_getInfo(self):
        name = self.jsonParser.getInfo(self.json_without_segmentation)
        info = '{"ts": 1642170799, "checksum": "SHA256:f1f8ef13433b1f0966e589818f3180750606eff69b3bc4a55e0181d7a9da8da1", "name": "6245968.dcm", "strategy": {"Random": {"ts": 1642371057, "client_id": "Dr Radiologist"}}}'
        infoDict = json.loads(info)
        self.assertEqual(name, infoDict)

    def test_isSegmented(self):
        name = self.jsonParser.isSegmented(self.json_without_segmentation)
        self.assertEqual(False, name)

    def test_getSegmentationName(self):
        name = self.jsonParser.getSegmentationName(self.json_without_segmentation)
        self.assertEqual(name, "6245968.seg.nrrd")

    def test_hasKeyAnnotate(self):
        result = self.jsonParser.hasKeyAnnotate(self.json_without_segmentation)
        self.assertEqual(result, False)

    def test_hasKeyRandom(self):
        result = self.jsonParser.hasKeyRandom(self.json_without_segmentation)
        self.assertEqual(result, True)

    def test_getClientId(self):
        result = self.jsonParser.getClientId(self.json_without_segmentation)
        self.assertEqual(result, "Dr Radiologist")

    def test_getMetaStatus(self):
        result = self.jsonParser.getMetaStatus("final", self.json_with_segmentation)
        self.assertEqual(result, "flagged")

    def test_getMetaLevel(self):
        result = self.jsonParser.getMetaLevel("final", self.json_with_segmentation)
        self.assertEqual(result, "easy")

    def test_getMetaApprovedBy(self):
        result = self.jsonParser.getMetaApprovedBy("final", self.json_with_segmentation)
        self.assertEqual(result, "Prof Radiogolist")

    def test_getMetaEditTime(self):
        result = self.jsonParser.getMetaEditTime("final", self.json_with_segmentation)
        self.assertEqual(result, "Thu Jan 13 12:21:01 2022")

    def test_getMetaComment(self):
        result = self.jsonParser.getMetaComment("final", self.json_with_segmentation)
        self.assertEqual(result, "Segementation was not easy")

    def test_jsonToImageData(self):
        imageData: ImageData = self.jsonParser.jsonToImageData("6662775.dcm", self.json_with_segmentation)
        self.assertEqual(imageData.getName(), "6662775.dcm")
        self.assertEqual(imageData.getFileName(), "6662775.dcm")
        self.assertEqual(
            imageData.getCheckSum(), "SHA256:1b474d23bda3de0c28f4287a7c0380d461e9f71d0dc468e64f336412cb575327"
        )
        self.assertEqual(imageData.isSegemented(), True)
        self.assertEqual(imageData.getClientId(), "Annotator")
        self.assertEqual(imageData.getStatus(), "flagged")
        self.assertEqual(imageData.getLevel(), "easy")
        self.assertEqual(imageData.getApprovedBy(), "Prof Radiogolist")
        self.assertEqual(imageData.getTimeOfAnnotation(), "2021-12-16 10:35:51")

    def test_extractLabelNames(self):
        labelsDict = self.jsonParser.extractLabels(self.json_with_multiple_versions)
        labelNames = self.jsonParser.extractLabelNames(labelsDict)
        self.assertListEqual(labelNames, ["final", "version_1", "version_2", "version_3", "version_4"])

    def test_jsonToImageData_with_multiple_versions(self):
        imageData: ImageData = self.jsonParser.jsonToImageData("lan.dcm", self.json_with_multiple_versions)

        self.assertEqual(imageData.getName(), "lan.dcm")
        self.assertEqual(imageData.getFileName(), "lan.dcm")
        self.assertEqual(
            imageData.getCheckSum(), "SHA256:a48c454592a36c1d2895322320e5ab5479eb5e93d9d4f3e16825625033875d6f"
        )

        self.assertEqual(imageData.isSegemented(), True)
        self.assertEqual(imageData.getClientId(), "user-xyz")
        self.assertEqual(imageData.getTimeOfAnnotation(), "2022-01-02 17:52:47")
        self.assertListEqual(imageData.getVersionNames(), ["final", "version_1", "version_2", "version_3", "version_4"])
        # dictMeta = imageData.getsegmentationMetaDict()
        # for k,v in dictMeta.items():
        #     print("------- key: ", k)
        #     v.display()

    def test_extractLabelContentByName(self):
        labelsDict = self.jsonParser.extractLabels(self.json_with_multiple_versions)
        labelContent = self.jsonParser.extractLabelContentByName(labelsDict)
        exspectedLabelContent = {
            "label_info": [
                {"name": "Lung", "idx": 1},
                {"name": "Heart", "idx": 2},
                {"name": "Trachea", "idx": 3},
                {"name": "Mediastinum", "idx": 4},
                {"name": "Clavicle", "idx": 5},
            ]
        }
        self.assertDictEqual(exspectedLabelContent, labelContent)

    def test_extractSegmentationMetaOfVersion(self):
        labelsDict = self.jsonParser.extractLabels(self.json_with_multiple_versions)
        labelContent = self.jsonParser.extractSegmentationMetaOfVersion(labelsDict, labelName="version_3")
        segmentationMeta = self.jsonParser.produceSegementationData(labelContent)

        self.assertEqual("self.status_3", segmentationMeta.getStatus())
        self.assertEqual("self.level_3", segmentationMeta.getLevel())
        self.assertEqual("self.approvedBy_3", segmentationMeta.getApprovedBy())
        self.assertEqual("self.comment_3", segmentationMeta.getComment())
        self.assertEqual(1656312200, segmentationMeta.getEditTime())

    def test_extractSegmentationMetaOfVersion_final_as_label(self):
        labelsDict = self.jsonParser.extractLabels(self.json_with_multiple_versions)
        labelContent = self.jsonParser.extractSegmentationMetaOfVersion(labelsDict, labelName="final")
        segmentationMeta = self.jsonParser.produceSegementationData(labelContent)

        self.assertEqual("self.status_final", segmentationMeta.getStatus())
        self.assertEqual("self.level_final", segmentationMeta.getLevel())
        self.assertEqual("self.approvedBy_final", segmentationMeta.getApprovedBy())
        self.assertEqual("self.comment_final", segmentationMeta.getComment())
        self.assertEqual(1656312100, segmentationMeta.getEditTime())

    def test_getAllSegmentationMetaOfAllLabels(self):
        labelsDict = self.jsonParser.extractLabels(self.json_with_multiple_versions)
        labelNames = self.jsonParser.extractLabelNames(labelsDict)
        segmentationMetaDict = self.jsonParser.getAllSegmentationMetaOfAllLabels(labelsDict, labelNames)

        self.assertNotIn("version_2", segmentationMetaDict)
        self.assertListEqual(list(segmentationMetaDict.keys()), ["final", "version_1", "version_3", "version_4"])

        self.assertEqual("self.status_final", segmentationMetaDict["final"].getStatus())
        self.assertEqual("self.level_final", segmentationMetaDict["final"].getLevel())
        self.assertEqual("self.approvedBy_final", segmentationMetaDict["final"].getApprovedBy())
        self.assertEqual("self.comment_final", segmentationMetaDict["final"].getComment())
        self.assertEqual(1656312100, segmentationMetaDict["final"].getEditTime())

        self.assertEqual("self.status_1", segmentationMetaDict["version_1"].getStatus())
        self.assertEqual("self.level_1", segmentationMetaDict["version_1"].getLevel())
        self.assertEqual("self.approvedBy_1", segmentationMetaDict["version_1"].getApprovedBy())
        self.assertEqual("self.comment_1", segmentationMetaDict["version_1"].getComment())
        self.assertEqual(1656312180, segmentationMetaDict["version_1"].getEditTime())

        self.assertEqual("self.status_3", segmentationMetaDict["version_3"].getStatus())
        self.assertEqual("self.level_3", segmentationMetaDict["version_3"].getLevel())
        self.assertEqual("self.approvedBy_3", segmentationMetaDict["version_3"].getApprovedBy())
        self.assertEqual("self.comment_3", segmentationMetaDict["version_3"].getComment())
        self.assertEqual(1656312200, segmentationMetaDict["version_3"].getEditTime())

        self.assertEqual("approved", segmentationMetaDict["version_4"].getStatus())
        self.assertEqual("self.level_4", segmentationMetaDict["version_4"].getLevel())
        self.assertEqual("self.approvedBy_4", segmentationMetaDict["version_4"].getApprovedBy())
        self.assertEqual("self.comment_4", segmentationMetaDict["version_4"].getComment())
        self.assertEqual(1656312200, segmentationMetaDict["version_4"].getEditTime())


if __name__ == "__main__":
    unittest.main()
