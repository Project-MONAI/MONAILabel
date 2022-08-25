.. comment
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


==========
What's New
==========

0.4.2
=====
- MONAI Bundle App - Pull `compatible <https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/monaibundle>`_ bundles from `MONAI Zoo <https://github.com/Project-MONAI/model-zoo>`_

  - spleen_ct_segmentation
  - spleen_deepedit_annotation
  - others
- Support for MONAI `0.9.1 <https://github.com/Project-MONAI/MONAI/releases/tag/0.9.1>`_



0.4.1
=====
- Fix MONAI dependency version to 0.9.0



0.4.0
=====
- Pathology Sample App

  - DeepEdit, Segmentation, NuClick models
  - Digital Slide Archive plugin
  - QuPath plugin
- Histogram-based GraphCut and Gaussian Mixture Model (GMM) based methods for scribbles

- Support for MONAI (supports 0.9.0 and above)
- Radiology Sample App (Aggregation of previous radiology models)
  - DeepEdit, Deepgrow, Segmentation, SegmentationSpleen models
- NrrdWriter for multi-channel arrays
- 3D Slicer Fixes

  - Support Segmentation Editor and other UI enhancements
  - Improvements for Scribble Interactions
  - Support for .seg.nrrd segmentation files
  - Support to pre-load existing label masks during image fetch/load
- Static checks using pre-commit ci



0.3.0
=====
- Multi GPU support for training

  - Support for both Windows and Ubuntu
  - Option to customize GPU selection
- Multi Label support for DeepEdit

  - DynUNET and UNETR
- Multi Label support for Deepgrow App

  - Annotate multiple organs (spleen, liver, pancreas, unknown etc..)
  - Train Deepgrow 2D/3D models to learn on existing + new labels submitted
- 3D Slicer plugin

  - Multi Label Interaction
  - UI Enhancements
  - Train/Update specific model
- Performance Improvements

  - Dataset (Cached, Persistence, SmartCache)
  - ThreadDataloader
  - Early Stopping
- Strategy Improvements to support Multi User environment
- Extensibility for Server APIs

0.2.0
=====

- Support for DICOMWeb connectivity to PACS `➔ <quickstart.html#setup-development-dicom-server>`__
- Annotations support via OHIF UI enabled in MONAI Label Server `➔ <quickstart.html#deepedit-annotation-in-ohif>`__
- Support for native and custom scoring methods to support next image selection strategies `➔ <modules.html#image-selection-strategy>`__

  - Native support for scoring and image selection using Epistemic Uncertainty and Test-time Augmentations (Aleatoric Uncertainty)

- Scribbles-based annotation support for all sample apps
- Simplified sample apps with default behavior for generic annotation tasks
