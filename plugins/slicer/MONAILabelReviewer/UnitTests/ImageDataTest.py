
import unittest
import sys
import json
import os

sys.path.append("..")
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.SegmentationMeta import SegmentationMeta
from MONAILabelReviewerLib.JsonParser import JsonParser

class ImageDataTest(unittest.TestCase):
    
    
    @classmethod
    def setUp(self):
        self.jsonParser = JsonParser(None)
        self.imageData = self.createTestImageData()
        self.parsedImageData = self.parseJsonToImageData()

    @classmethod
    def parseJsonToImageData(self) -> ImageData:
        fileNameWithMultipleVersions = "test_datastore_v2_image_with_multiple_versions.json"
        fileNameWithMultipleVersions_str = self.loadJsonStr(fileNameWithMultipleVersions)
        json_with_multiple_versions= json.loads(fileNameWithMultipleVersions_str)
        return self.jsonParser.jsonToImageData("lan.dcm", json_with_multiple_versions)

    @classmethod
    def loadJsonStr(self, fileName: str) -> str:
        with open(os.path.join(sys.path[0], "TestDataSet/" + fileName)) as f:
            data = json.dumps(json.load(f))
        return data


    @classmethod
    def createTestImageData(self) -> ImageData:
        name = "6667571"
        fileName =  "6667571.dcm"
        nodeName =  "6667571.dcm"
        checkSum =  "SHA256:2a454e9ab8a33dc74996784163a362a53e04adcee2fd73a8b6299bf0ce5060d3"
        isSegmented =  True
        timeStamp =  1639647550
        comment = "test-comment"

        imageData = ImageData(name=name, fileName=fileName, nodeName=nodeName, checkSum=checkSum, segmented = isSegmented, timeStamp=timeStamp, comment=comment)
        
        #imageData.addNewSegmentationMeta(status="flagged", level="hard", approvedBy="Dr.Faust", comment="Damit ich erkenne, was die Menchenwelt im Innerstern zusammenhaelt")
        imageData.setSegmentationFileName("6662775.seg.nrrd")
        imageData.setClientId("segmentator")
        imageData.setVersionNames(["final", "version_1", "version_2"])

        segMeta_final = self.createSegMeta(status="flagged", level="hard", approvedBy="annotator", comment="Errare humanum est", editTime="Thu Jan 13 12:21:01 2022")
        segMeta_version1 = self.createSegMeta(status="flagged", level="medium", approvedBy="radiologist_1", comment="Irren ist menschlisch", editTime="Thu Jan 14 12:21:01 2022")
        segMeta_version2 = self.createSegMeta(status="flagged", level="easy", approvedBy="radiologist_2", comment="Menschlische Gebrechen sühnet reine Menschlichkeit", editTime="Thu Jan 15 12:21:01 2022")

        segmentationMetaDict = {}
        segmentationMetaDict["final"] = segMeta_final
        segmentationMetaDict["version_1"] = segMeta_version1
        segmentationMetaDict["version_2"] = segMeta_version2

        imageData.setSegmentationMetaDict(segmentationMetaDict)

        return imageData

    @classmethod
    def createSegMeta(self, status, level, approvedBy, comment, editTime) -> SegmentationMeta:
        segmentationMeta = SegmentationMeta()
        segmentationMeta.build(status=status, level=level, approvedBy=approvedBy, comment=comment, editTime=editTime)
        return segmentationMeta


    def test_getStatus(self):
        status = self.imageData.getStatus(versionTag = "final")
        self.assertEqual(status, "flagged")

    def test_getLatestVersionTag(self):
        latestVersion = self.imageData.getLatestVersionTag()
        self.assertEqual(latestVersion, "version_2")

    def test_getLatestVersionTag_remove_last_version(self):
        self.imageData.deleteVersionName("version_2")
        versionNames = self.imageData.getVersionNames()
        self.assertListEqual(['final', 'version_1'], versionNames)

    def test_getLatestVersionTag_remove_version_inbetween(self):
        self.imageData.deleteVersionName("version_1")
        versionNames = self.imageData.getVersionNames()
        self.assertListEqual(['final', 'version_2'], versionNames)

    def test_getMetaByVersionTag(self):
        segMeta_version2 = self.imageData.getMetaByVersionTag("version_2")
        
        exspectMeta = {
            'segmentationMeta': {
                'status': 'flagged', 
                'approvedBy': 'radiologist_2',
                'level': 'easy', 
                'comment': 'Menschlische Gebrechen sühnet reine Menschlichkeit', 
                'editTime': 'Thu Jan 15 12:21:01 2022'
                }
            }

        self.assertDictEqual(exspectMeta, segMeta_version2)
        

    def test_obtainUpdatedParams(self):
        params = self.parsedImageData.obtainUpdatedParams("version_3")
        exspectedParams = {
                            'label_info': 
                                [
                                    {'name': 'Lung', 'idx': 1}, 
                                    {'name': 'Heart', 'idx': 2}, 
                                    {'name': 'Trachea', 'idx': 3}, 
                                    {'name': 'Mediastinum', 'idx': 4},
                                    {'name': 'Clavicle', 'idx': 5}
                                ],
                            'segmentationMeta': {
                                'status': 'self.status_3', 
                                'approvedBy': 'self.approvedBy_3', 
                                'level': 'self.level_3', 
                                'comment': 'self.comment_3', 
                                'editTime': 1656312200}
                            }
        self.assertDictEqual(exspectedParams, params)

    def test_isEqualSegmentationMeta(self):
        isEqual = self.parsedImageData.isEqualSegmentationMeta(tag = "version_3", status = "self.status_3", level  = "self.level_3", approvedBy  = "self.approvedBy_3", comment  = "self.comment_3")
        self.assertTrue(isEqual)

    def test_isEqualSegmentationMeta_when_segmentationmetadata_does_not_exit_add(self):
        
        isEqual = self.parsedImageData.isEqualSegmentationMeta(tag = "version_4", status = "self.status_4", level  = "self.level_4", approvedBy  = "self.approvedBy_4", comment  = "self.comment_4")
        
        self.assertFalse(isEqual)
        metas = self.parsedImageData.getsegmentationMetaDict()
        self.assertTrue("version_4" in metas)
        
    def test_getClientId_when_request_init_segmentation(self):
        clientId = self.parsedImageData.getClientId("final")
        self.assertEqual('user-xyz', clientId)

    def test_getClientId_when_request_edit_version(self):
        clientId = self.parsedImageData.getClientId("version_3")
        self.assertEqual('user-xyz', clientId)

    def test_getComment_when_request_init_segmentation(self):
        comment = self.parsedImageData.getComment("final")
        self.assertEqual('self.comment_final', comment)

    def test_getComment_when_request_edit_version(self):
        comment = self.parsedImageData.getComment("version_3")
        self.assertEqual('self.comment_3', comment)

    def test_getApprovedBy_when_request_init_segmentation(self):
        approvedBy = self.parsedImageData.getApprovedBy("final")
        self.assertEqual('self.approvedBy_final', approvedBy)

    def test_getApprovedBy_when_request_edit_version(self):
        approvedBy = self.parsedImageData.getApprovedBy("version_3")
        self.assertEqual('self.approvedBy_3', approvedBy)

    def test_isFlagged_when_segmentation_is_not_flagged(self):
        isFlagged = self.parsedImageData.isFlagged("version_3")
        self.assertFalse(isFlagged)

    def test_isFlagged_when_segmentation_is_flagged(self):
        isFlagged = self.imageData.isFlagged("version_1")
        self.assertTrue(isFlagged)

    def test_hasSegmentationMeta_when_has_segmentation(self):
        hasSegmentationMeta = self.imageData.hasSegmentationMeta("version_1")
        self.assertTrue(hasSegmentationMeta)

    def test_hasSegmentationMeta_when_version_does_not_exit(self):
        hasSegmentationMeta = self.imageData.hasSegmentationMeta("version_4")
        self.assertFalse(hasSegmentationMeta)


    def test_getTimeOfEditing(self):
        editTime = self.parsedImageData.getTimeOfEditing("version_1")
        self.assertEqual("2022-06-27 08:43:00", editTime)

        # for k,v in metas.items():
        #     print("key: ", k)
        #     v.display()

    def test_getSegmentationMetaByVersionTag(self):
        segmentationData : SegmentationMeta =  self.parsedImageData.getSegmentationMetaByVersionTag("version_1")
        self.assertEqual("self.status_1", segmentationData.getStatus())
        self.assertEqual("self.level_1", segmentationData.getLevel())
        self.assertEqual("self.approvedBy_1", segmentationData.getApprovedBy())
        self.assertEqual(1656312180, segmentationData.getEditTime())
        self.assertEqual("self.comment_1", segmentationData.getComment())

    def test_isApproved(self):
        isApproved =  self.parsedImageData.isApproved()
        print(isApproved)







if __name__ == "__main__":
    unittest.main()