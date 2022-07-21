
import unittest
import sys

sys.path.append("..")
from MONAILabelReviewerLib.ImageData import ImageData

class ImageDataTest(unittest.TestCase):
    
    
    @classmethod
    def setUp(self):
        name = "6667571"
        fileName =  "6667571.dcm"
        nodeName =  "6667571.dcm"
        checkSum =  "SHA256:2a454e9ab8a33dc74996784163a362a53e04adcee2fd73a8b6299bf0ce5060d3"
        isSegmented =  True
        timeStamp =  1639647550
        comment = "test-comment"


        self.imageData = ImageData(name=name, fileName=fileName, nodeName=nodeName, checkSum=checkSum, segmented = isSegmented, timeStamp=timeStamp, comment=comment)
        
        self.imageData.setSegmentationMeta(status="flagged", level="hard", approvedBy="Dr.Faust", comment="Damit ich erkenne, was die Menchenwelt im Innerstern zusammenhaelt")
        self.imageData.setSegmentationFileName("6662775.seg.nrrd")
        self.imageData.setClientId("segmentator")
        self.imageData.setVersionNames(["final", "version_1", "version_2"])

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


if __name__ == "__main__":
    unittest.main()