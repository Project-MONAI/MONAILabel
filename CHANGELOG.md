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

# Changelog
All notable changes to MONAILabel are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.6.0] - 2022-12-19
### Added
* Pathology Models
  * [NuClick annotation model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/pathology_nuclick_annotation)
  * [Nuclei classification model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/pathology_nuclei_classification)
  * [HoverNet model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/pathology_nuclei_segmentation_classification)
* QuPath Extension: [0.3.1](https://github.com/Project-MONAI/MONAILabel/releases/download/0.6.0/qupath-extension-monailabel-0.3.1.jar)
  * User experience enhancements
  * MONAI Label specific Toolbar actions
  * Drag and Drop ROI to run auto-segmentation models
  * Single click to run interaction models (NuClick)
  * Support Next Sample/ROI for Active Learning
* Experiment Management
* 3D Slicer: Detection model support in MONAI Bundle App for Radiology use-case
* Multi-GPU/Multi-Threaded support for Batch Inference
### Changed
* Support the latest version of bundles in MONAI Bundle App
* Upgrade vertebra pipeline
* MONAI version >= 1.1.0
### Removed
* DeepEdit model for Nuclei segmentation in Pathology

## [0.5.2] - 2022-10-24
### Added
* Bundle support (Endoscopy Sample App)
  * [Tool Tracking segmentation model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/endoscopic_tool_segmentation)
  * [InBody vs OutBody classification model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/endoscopic_inbody_classification)
* Rest APIs to download the latest trained model and stats
* Interactive NuClick segmentation in DSA | [Demo](https://medicine.ai.uky.edu/wp-content/uploads/2022/10/interactive_cell_labeling_via_nucklick_in_dsa.mp4)
### Changed
* Support for MONAI [1.0.1](https://github.com/Project-MONAI/MONAI/releases/tag/1.0.1)
* Option to disable DICOM to NIFTI conversion for OHIF usecase
* Proxy URL fixes for GCP + DICOMWeb support
* 3D Slicer UI Improvements
  * [Optimize window size for options/config](https://user-images.githubusercontent.com/7339051/194677198-d9deb3f2-a728-453a-b68d-a1b21afa6bee.png)
  * Download label feature to fetch original labels
* Improvements on MONAI Bundle App
  * support local bundles (pre-downloaded)
  * support customized scripts
### Fixed
* Fixes for multi label output of `segmentation_nuclei` model (Pathology)
### Removed
* Remove option to run ALL Training Tasks in 3D Slicer

## [0.5.1] - 2022-09-16
### Added
* Endoscopy Sample App
  * Tool Tracking segmentation model | [Demo](https://drive.google.com/file/d/190rqvSMQULUlzS3XDAZVfPma1_6znPsd/view?usp=sharing)
  * InBody vs OutBody classification model | [Demo](https://drive.google.com/file/d/1Ii3_mYvHVykC-UsytdFvvPdfG1oIqz8u/view?usp=sharing)
  * DeepEdit interaction model
  * CVAT Integration to support automated workflow to run Active Learning Iterations
* Multi Stage vertebra segmentation in Radiology App
### Changed
* Improving performance for Radiology App
  * Support cache for pre-transforms in case repeated inference for interaction models
  * Support cache for DICOM Web API responses
  * Optimize pre transforms to run GPU for max throughput
  * DICOM Proxy for wado/qido
* Improvements for Epistemic (v2) active learning strategy
* Support for MONAI 1.0.0 and above
### Fixed
* Scribbles to support MetaTensor
### Deprecated
* TTA Based active learning strategy is deprecated

## [0.4.2] - 2021-07-25
### Added
* [MONAI Bundle App](sample-apps/monaibundle) - Pull compatible bundles from [MONAI Zoo](https://github.com/Project-MONAI/model-zoo)
  * spleen_ct_segmentation
  * spleen_deepedit_annotation
  * others
### Changed
* Support for MONAI [0.9.1](https://github.com/Project-MONAI/MONAI/releases/tag/0.9.1)

## [0.4.1] - 2021-07-05
### Changed
* MONAI dependency version to 0.9.0

## [0.4.0] - 2022-06-13
### Added
* Pathology Sample App
  * DeepEdit, Segmentation, [NuClick](https://arxiv.org/abs/2005.14511) models
  * Digital Slide Archive plugin | [Demo](https://drive.google.com/file/d/16HnQY81kAVEbD9TvhAp_hlLnfgHQgX8I/view)
  * QuPath plugin | [Demo](https://drive.google.com/file/d/18mQ5DXuThp9YxXcbR0f19yS2klhmZozG/view)
* Histogram-based GraphCut and Gaussian Mixture Model (GMM) based methods for scribbles
### Changed
* Support for MONAI (supports 0.9.0 and above)
* Radiology Sample App (Aggregation of previous radiology models)
  * DeepEdit, Deepgrow, Segmentation, SegmentationSpleen models
* NrrdWriter for multi-channel arrays
* Static checks using pre-commit ci
### Fixed
* 3D Slicer
  * Support Segmentation Editor and other UI enhancements
  * Improvements for Scribble Interactions
  * Support for _**.seg.nrrd**_ segmentation files
  * Support to pre-load existing label masks during image fetch/load
### Removed
* SimpleCRF and dependent functions for scribbles

## [0.3.2] - 2022-02-28
### Added
* SSL and multiple worker options while starting MONAI Label server
* Scribbles support for OHIF Viewer
* Add Citation page
### Changed
* Support for MONAI (supports 0.8.1 and above)
* Upgrade PIP Dependencies
### Fixed
* Load pretrained models for TTA and Epistemic Scoring
* Load MMAR API for deepgrow/segmentation apps
* Documentation Fixes

## [0.3.1] - 2021-12-29
### Added
* Flexible version support for MONAI (supports 0.8.* instead of 0.8.0)
### Changed
* Strict Flag is set to False while loading pretrained models
### Fixed
* Inverse transform for DeepEdit sample app
* Documentation Fixes

## [0.3.0] - 2021-11-28
### Added
* Multi GPU support for training
  * Support for both Windows and Ubuntu
  * Option to customize GPU selection
* Multi Label support for DeepEdit
  * DynUNET and UNETR
* Multi Label support for Deepgrow App
  * Annotate multiple organs (spleen, liver, pancreas, unknown etc..)
  * Train Deepgrow 2D/3D models to learn on existing + new labels submitted
* 3D Slicer plugin
  * Multi Label Interaction
  * UI Enhancements
  * Train/Update specific model
* Performance Improvements
  * Dataset (Cached, Persistence, SmartCache)
  * ThreadDataloader
  * Early Stopping
* Strategy Improvements to support Multi User environment
* Extensibility for Server APIs
### Changed
* Operate histogram likelihood transform in both normalized and unnormalized modes for Scribbles
### Removed
* DeepGrow Left-Atrium

## [0.2.0] - 2021-09-23
### Added
* Support for DICOMWeb connectivity to local/remote PACS
* Annotations support via OHIF UI enabled in MONAI Label Server
* Support for native and custom scoring methods to support next image selection strategies
  * Native support for scoring and image selection using Epistemic Uncertainty and Test-time Augmentations (Aleatoric Uncertainty)
* Custom `ScoringMethod` and `Strategy` implementation documentation
* Scribbles-based annotation support for all sample apps
### Changed
* Previously named `generic` apps now have default functionality under `deepedit`, `deepgrow` and `segmentation`
* Updated `Modules Overview` documentation to include interaction between `ScoringMethod` and `Strategy`
### Removed
* All spleen segmentation sample apps (DeepGrow, DeepEdit, auto-segmentation)

## [0.1.0] - 2021-07-14
### Added
* Framework for developing and deploying MONAI Label Apps to train and infer AI models
* Compositional & portable APIs for ease of integration in existing workflows
* Customizable design for varying user expertise
* 3DSlicer support
* Support for multi-label auto-segmentation
* Template apps to customize the behavior of DeepGrow and DeepEdit
* Automated segmentation of left atrium, spleen
* DeepGrow AI annotation of left atrium, spleen
* DeepEdit AI annotation of left atrium, spleen