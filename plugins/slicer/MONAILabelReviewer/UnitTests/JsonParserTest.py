import json
import os
import sys
import unittest

sys.path.append("..")
from ReviewerLibs.ImageData import ImageData
from ReviewerLibs.JsonParser import JsonParser


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
        self.assertEqual(imageData.getTime(), "2021-12-16 10:35:51")


if __name__ == "__main__":
    unittest.main()
