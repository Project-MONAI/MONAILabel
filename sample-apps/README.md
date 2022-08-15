<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Available MONAI Label Apps

### Overview

In this folder you will find examples of the **three available paradigms** across Radiology and Pathology use-cases: Two interactive (DeepGrow, DeepEdit) and one for automated segmentation.

## [Radiology](./radiology)

This app has example models to do both interactive and automated segmentation over radiology (3D) images.
If you are developing any examples related to radiology, you should refer this app.  It has examples for following 3 types of model.
- DeepEdit (Interactive + Auto Segmentation)
  - Spleen, Liver, Kidney, etc...
- Deepgrow (Interactive)
  - Any Organ/Tissue but pretrained to work well on Spleen, Liver, Kidney, etc...
- Segmentation (Auto Segmentation)
  - Spleen, Liver, Kidney, etc...


## [Pathology](./pathology)

This app has example models to do both interactive and automated segmentation over pathology (WSI) images.
If you are developing any examples related to pathology, you should refer this app.  It has examples for following 2 types of model.
- DeepEdit (Interactive + Auto Segmentation)
  - Nuclei Segmentation
- Segmentation (Auto Segmentation)
  - Nuclei multi-label segmentation for
    - Neoplastic cells
    - Inflammatory
    - Connective/Soft tissue cells
    - Dead Cells
    - Epithelial


## [MONAI Bundle](./monaibundle)

This app has example models to do both interactive and automated segmentation using monai-bundles defined in [MONAI ZOO](https://github.com/Project-MONAI/model-zoo/tree/dev/models).
It can pull any bundle defined in the zoo if it is compatible and follows the checklist as defined [here](./monaibundle).
