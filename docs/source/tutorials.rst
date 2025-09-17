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

=========
Tutorials
=========

This section provides comprehensive tutorials for using MONAI Label with advanced models and viewers.

VISTA3D Universal Segmentation
===============================

VISTA3D is a universal foundation model for 3D CT segmentation that can segment 117+ anatomical structures 
including organs, bones, vessels, and soft tissues. It supports both automatic and interactive segmentation workflows.

**Quick Start**

For immediate setup and basic usage, see the `VISTA3D Quick Start Guide`_.

**Complete Tutorial** 

For comprehensive instructions including advanced configuration, troubleshooting, and clinical workflows, 
see the `VISTA3D Complete Tutorial`_.

**Configuration Examples**

For sample configurations covering different deployment scenarios (memory optimization, PACS integration, 
multi-model setup, etc.), see the `VISTA3D Configuration Examples`_.

Key Features
------------

- **Universal Segmentation**: Supports 117+ anatomical classes
- **Interactive Prompting**: Point-based and class-based interactive segmentation  
- **Multi-Viewer Support**: Works with OHIF and 3D Slicer
- **High Performance**: Optimized for clinical workflows
- **Flexible Deployment**: Local, cloud, and PACS integration options

Supported Viewers
-----------------

VISTA3D integration is available through:

- **OHIF Viewer**: Web-based DICOM viewer with full VISTA3D support
- **3D Slicer**: Desktop application with VISTA3D capabilities

Prerequisites
-------------

- Python 3.8+ with MONAI Label installed
- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)  
- CT data in DICOM or NIfTI format
- Compatible viewer (OHIF or 3D Slicer)

Basic Usage
-----------

1. **Setup**::

    # Download bundle app
    monailabel apps --download --name monaibundle --output apps
    
    # Start VISTA3D server  
    monailabel start_server \\
      --app apps/monaibundle \\
      --studies datasets/ct_volumes \\
      --conf models vista3d \\
      --conf preload true

2. **Automatic Segmentation**: Load CT volume and run auto-segmentation for 117+ structures

3. **Interactive Prompting**: Use point prompts to refine segmentation of specific anatomy  

4. **Class Selection**: Choose anatomical categories (organs, bones, vessels) for targeted segmentation

For detailed instructions, examples, and troubleshooting, refer to the tutorial links above.

Additional Resources
====================

For more MONAI Label tutorials covering other models and use cases, visit the 
`MONAI Tutorials Repository <https://github.com/Project-MONAI/tutorials/tree/main/monailabel>`_.

.. _VISTA3D Quick Start Guide: tutorials/vista3d_quickstart.md
.. _VISTA3D Complete Tutorial: tutorials/vista3d_tutorial.md  
.. _VISTA3D Configuration Examples: tutorials/vista3d_configurations.md