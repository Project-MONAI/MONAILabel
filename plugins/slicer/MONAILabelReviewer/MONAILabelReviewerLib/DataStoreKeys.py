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

from typing import List

"""
DataStoreKeys contains arrays which represent the structure of datastore.json
That provides an overview of keys which are relevant for the access the key-value-pair
within json file.
Below please have a look on an example of such entry in datastore.json.
"""


class DataStoreKeys:
    def __init__(self):

        self.OBJECT = "objects"
        self.FINAL = "final"
        self.ORIGINAL = "original"
        self.ANNOTATE = "annotate"
        self.RANDOM = "Random"
        self.INFO = "info"
        self.LABEL_INFO = "label_info"

        self.IMAGE_INFO = ["image", "info"]
        self.FILENAME = ["image", "info", "name"]
        self.NODE_NAME = ["image", "info", "name"]

        self.CHECKSUM = ["image", "info", "checksum"]
        self.TIMESTAMP = ["image", "info", "ts"]
        self.TIMESTAMP_ANNOTATE = ["image", "info", "strategy", "annotate", "ts"]
        self.TIMESTAMP_RANDOM = ["image", "info", "strategy", "Random", "ts"]

        self.STRATEGY = ["image", "info", "strategy"]
        self.CLIENT_ID_BY_ANNOTATE = ["image", "info", "strategy", "annotate", "client_id"]
        self.CLIENT_ID_BY_RANDOM = ["image", "info", "strategy", "Random", "client_id"]
        self.CLIENT_ID = ["labels", "final", "info", "client_id"]

        self.LABELS = ["labels"]
        self.LABELS_FINAL = ["labels", "final"]
        self.LABELS_FINAL_INFO = ["labels", "final", "info"]
        self.LABELS_INFO = ["labels", "original", "info", "label_info"]
        self.LABELS_FINAL_INFO_LABELS_INFO = ["labels", "final", "info", "label_info"]
        self.SEGMENTATION_NAME_BY_FINAL = ["labels", "final", "info", "name"]
        self.SEGMENTATION_NAME_BY_ORIGINAL = ["labels", "original", "info", "name"]

        # Additional entries in json file which contains meta data
        self.META = "segmentationMeta"
        self.META_STATUS = "status"
        self.META_LEVEL = "level"
        self.APPROVED_BY = "approvedBy"

        self.META_EDIT_TIME = "editTime"
        self.META_COMMENT = "comment"

    def getMeta(self, key: str, label: str) -> List[str]:
        return ["labels"] + [label] + ["info", "segmentationMeta"] + [key]

    def getMetaStatus(self, label: str):
        return self.getMeta(key=self.META_STATUS, label=label)

    def getMetaLevel(self, label: str):
        return self.getMeta(key=self.META_LEVEL, label=label)

    def getMetaApprovedBy(self, label: str):
        return self.getMeta(key=self.APPROVED_BY, label=label)

    def getMetaEditTime(self, label: str):
        return self.getMeta(key=self.META_EDIT_TIME, label=label)

    def getMetaComment(self, label: str):
        return self.getMeta(key=self.META_COMMENT, label=label)

    def getInfoInLabels(self, label: str):
        return ["labels"] + [label] + ["info"]
