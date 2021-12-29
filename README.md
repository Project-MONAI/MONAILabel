# MONAI Label

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/MONAILabel/workflows/build/badge.svg?branch=main)](https://github.com/Project-MONAI/MONAILabel/commits/main)
[![Documentation Status](https://readthedocs.org/projects/monailabel/badge/?version=latest)](https://docs.monai.io/projects/label/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/monailabel.svg)](https://badge.fury.io/py/monailabel)
[![Azure DevOps tests (compact)](https://img.shields.io/azure-devops/tests/projectmonai/monai-label/10?compact_message)](https://dev.azure.com/projectmonai/monai-label/_test/analytics?definitionId=10&contextType=build)
[![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/projectmonai/monai-label/10)](https://dev.azure.com/projectmonai/monai-label/_build?definitionId=10)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAILabel/branch/main/graph/badge.svg)](https://codecov.io/gh/Project-MONAI/MONAILabel)

MONAI Label is a server-client system that facilitates interactive medical image annotation by using AI. It is an
open-source and easy-to-install ecosystem that can run locally on a machine with single or multiple GPUs. Both server and client
work on the same/different machine. It shares the same principles with [MONAI](https://github.com/Project-MONAI).

[MONAI Label Demo](https://youtu.be/o8HipCgSZIw?t=1319)

![DEMO](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/demo.png)

## Features

> _The codebase is currently under active development._

- Framework for developing and deploying MONAI Label Apps to train and infer AI models
- Compositional & portable APIs for ease of integration in existing workflows
- Customizable labelling app design for varying user expertise
- Annotation support via 3DSlicer & OHIF 
- PACS connectivity via DICOMWeb

## Installation

MONAI Label supports following OS with **GPU/CUDA** enabled.

- Ubuntu
- [Windows](https://docs.monai.io/projects/label/en/latest/installation.html#windows)

To install the [current release](https://pypi.org/project/monailabel/), you can simply run:

```bash
  pip install monailabel
  
  # download sample apps/dataset
  monailabel apps --download --name deepedit --output apps
  monailabel datasets --download --name Task09_Spleen --output datasets
  
  # run server
  monailabel start_server --app apps/deepedit --studies datasets/Task09_Spleen/imagesTr
```

> If monailabel install path is not automatically determined, then you can provide explicit install path as: 
> 
> `monailabel apps --prefix ~/.local`

For **_prerequisites_**, other installation methods (using the default GitHub branch, using Docker, etc.), please refer
to the [installation guide](https://docs.monai.io/projects/label/en/latest/installation.html).

> Once you start the MONAI Label Server, by default server will be up and serving at http://127.0.0.1:8000/. Open the serving URL in browser. It will provide you the list of Rest APIs available. **For this, please make sure you use the HTTP protocol. HTTPS is not implemented.**

### 3D Slicer

Download **Preview Release** from https://download.slicer.org/ and install MONAI Label plugin from Slicer Extension
Manager.

Refer [3D Slicer plugin](plugins/slicer) for other options to install and run MONAI Label plugin in 3D Slicer.
> To avoid accidentally using an older Slicer version, you may want to _uninstall_ any previously installed 3D Slicer package.

### OHIF

MONAI Label comes with [pre-built plugin](plugins/ohif) for [OHIF Viewer](https://github.com/OHIF/Viewers).  To use OHIF Viewer, you need to provide DICOMWeb instead of FileSystem as _studies_ when you start the server.
> Please install [Orthanc](https://www.orthanc-server.com/download.php) before using OHIF Viewer.
> For Ubuntu 20.x, Orthanc can be installed as `apt-get install orthanc orthanc-dicomweb`. However, you have to **upgrade to latest version** by following steps mentioned [here](https://book.orthanc-server.com/users/debian-packages.html#replacing-the-package-from-the-service-by-the-lsb-binaries)
>
> You can use [PlastiMatch](https://plastimatch.org/plastimatch.html#plastimatch-convert) to convert NIFTI to DICOM

```bash
  # start server using DICOMWeb
  monailabel start_server --app apps\deepedit --studies http://127.0.0.1:8042/dicom-web
```

> OHIF Viewer will be accessible at http://127.0.0.1:8000/ohif/

![OHIF](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/ohif.png)

> **_NOTE:_** OHIF does not yet support Scribbles-based annotations and Multi-Label interaction for DeepEdit.

## Contributing

For guidance on making a contribution to MONAI Label, see the [contributing guidelines](CONTRIBUTING.md).

## Community

Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join
our [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

Ask and answer questions over
on [MONAI Label's GitHub Discussions tab](https://github.com/Project-MONAI/MONAILabel/discussions).

## Links

- Website: https://monai.io/
- API documentation: https://docs.monai.io/projects/label
- Code: https://github.com/Project-MONAI/MONAILabel
- Project tracker: https://github.com/Project-MONAI/MONAILabel/projects
- Issue tracker: https://github.com/Project-MONAI/MONAILabel/issues
- Wiki: https://github.com/Project-MONAI/MONAILabel/wiki
- Test status: https://github.com/Project-MONAI/MONAILabel/actions
- PyPI package: https://pypi.org/project/monailabel/
- Weekly previews: https://pypi.org/project/monailabel-weekly/
- Docker Hub: https://hub.docker.com/r/projectmonai/monailabel
