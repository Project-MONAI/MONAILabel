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

"""
This constant is a list maintaining bundle models that can be used in MONAI Label.
    MONAI Label supports most labeling models presented in MODEL ZOO, however, some bundles are not labeling tasks.
    For instance, MedNISTGAN, pathology tumor, valve_landmarks, etc are not suitable for MONAI Label.
    The MONAI Label team maintains and verifies usable bundles for MONAI Label support.
Get the latest version, as well as all existing versions of a bundle that is stored in the release of specified
    repository with the provided tag. https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1

The list is updating upon latest version of MODEL ZOO.

Note: Version tags are not needed.
"""

MAINTAINED_BUNDLES = [
    "pancreas_ct_dints_segmentation",  # Pancreas segmentation with radiology/monaibundle app support. Added Oct 2022
    "prostate_mri_anatomy",  # prostate segmentation with radiology/monaibundle app support. Added Oct 2022
    "renalStructures_UNEST_segmentation",  # renal cortex, medulla, pelvis segmentation with radiology/monaibundle support. Added Oct 2022
    "spleen_ct_segmentation",  # spleen segmentation with radiology/monaibundle support. Added Oct 2022
    "spleen_deepedit_annotation",  # spleen deepedit model annotation for CT images. Added Oct 2022
    "swin_unetr_btcv_segmentation",  # 3D transformer model for multi-organ segmentation. Added Oct 2022
    "wholeBrainSeg_Large_UNEST_segmentation",  # whole brain segmentation for T1 MRI brain images. Added Oct 2022
    "lung_nodule_ct_detection",  # The first lung nodule detection task can be used for MONAI Label. Added Dec 2022
    "wholeBody_ct_segmentation",  # The SegResNet trained TotalSegmentator dataset with 104 tissues. Added Feb 2023
]
