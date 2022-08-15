class ImageDataStatistics:
    def __init__(self):
        self.segmentationProgress : int = 0
        self.idxTotalSegmented : str = ""
        self.idxTotalApproved : str = ""
        self.progressPercentage : int = 0

        self.segmentationProgressAllPercentage : int = 0
        self.approvalProgressPercentage : int = 0


    def build(self, segmentationProgress = 0 , idxTotalSegmented = "", idxTotalApproved = "", progressPercentage  = 0, segmentationProgressAllPercentage = 0, approvalProgressPercentage = 0): 
        self.segmentationProgress = segmentationProgress
        self.idxTotalSegmented = idxTotalSegmented
        self.idxTotalApproved = idxTotalApproved
        self.progressPercentage = progressPercentage
        self.segmentationProgressAllPercentage = segmentationProgressAllPercentage
        self.approvalProgressPercentage = approvalProgressPercentage


    def getSegmentationProgress(self) -> int:
        return self.segmentationProgress

    def getIdxTotalSegmented(self) -> str:
        return self.idxTotalSegmented

    def getIdxTotalApproved(self) -> str:
        return self.idxTotalApproved

    def getProgressPercentage(self) -> int:
        return self.progressPercentage

    def getSegmentationProgressAllPercentage(self) -> int:
        return self.segmentationProgressAllPercentage

    def getApprovalProgressPercentage(self) -> int:
        return self.approvalProgressPercentage