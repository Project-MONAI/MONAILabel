# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ATTRB_MONAILABELINDICATOR = "DFE10001"
ATTRB_MONAILABELTAG = "DFE10002"
ATTRB_MONAILABELINFO = "DFE10003"

ATTRB_PATIENTID = "00100020"
ATTRB_STUDYINSTANCEUID = "0020000D"
ATTRB_SERIESINSTANCEUID = "0020000E"
ATTRB_SOPINSTANCEUID = "00080018"
ATTRB_SOPCLASSUID = "00020002"
ATTRB_IMPLCLASSUID = "00020012"
ATTRB_MODALITY = "00080060"  # header to check for DICOM SEG
ATTRB_REFERENCEDSERIESSEQUENCE = "00081115"


DICOMSEG_MODALITY = "SEG"
DICOMSEG_SOPCLASSUID = "1.2.840.10008.5.1.4.1.1.66.4"
DICOMSEG_IMPLCLASSUID = "1.2.840.10008.5.1.4.1.1.66.4"


def str2hex(input_string: str):
    return hex(int(input_string, 16))
