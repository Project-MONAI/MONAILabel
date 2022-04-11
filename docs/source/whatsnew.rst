==========
What's New
==========

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
