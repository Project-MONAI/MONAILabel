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


class ImageDataStatistics:
    def __init__(self):
        self.segmentationProgress: int = 0
        self.idxTotalSegmented: str = ""
        self.idxTotalApproved: str = ""
        self.progressPercentage: int = 0

        self.segmentationProgressAllPercentage: int = 0
        self.approvalProgressPercentage: int = 0

    def build(
        self,
        segmentationProgress=0,
        idxTotalSegmented="",
        idxTotalApproved="",
        progressPercentage=0,
        segmentationProgressAllPercentage=0,
        approvalProgressPercentage=0,
    ):
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
