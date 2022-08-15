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

# MONAI Label

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/MONAILabel/workflows/build/badge.svg?branch=main)](https://github.com/Project-MONAI/MONAILabel/commits/main)
[![Documentation Status](https://readthedocs.org/projects/monailabel/badge/?version=latest)](https://docs.monai.io/projects/label/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/monailabel.svg)](https://badge.fury.io/py/monailabel)
[![Azure DevOps tests (compact)](https://img.shields.io/azure-devops/tests/projectmonai/monai-label/10?compact_message)](https://dev.azure.com/projectmonai/monai-label/_test/analytics?definitionId=10&contextType=build)
[![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/projectmonai/monai-label/10)](https://dev.azure.com/projectmonai/monai-label/_build?definitionId=10)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAILabel/branch/main/graph/badge.svg)](https://codecov.io/gh/Project-MONAI/MONAILabel)

MONAI Label is a server-client system that facilitates interactive medical image annotation by using AI. It is an
open-source and easy-to-install ecosystem that can run locally on a machine with single or multiple GPUs. Both server
and client work on the same/different machine. It shares the same principles
with [MONAI](https://github.com/Project-MONAI).

[MONAI Label](https://youtu.be/m2rYorVwXk4) | [Demo](https://youtu.be/o8HipCgSZIw?t=1319)


![DEMO](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/demo.png)
<table>
<tr>
<td><img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/ohif.png" alt="drawing" width="150"/></td>
<td><img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/dsa.jpg" alt="drawing" width="150"/></td>
<td><img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/qupath.jpg" alt="drawing" width="150"/></td>
<td><img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/cvat_detector.jpeg" alt="drawing" width="150"/></td>
</tr>
</table>

## Features

> _The codebase is currently under active development._

- Framework for developing and deploying MONAI Label Apps to train and infer AI models
- Compositional & portable APIs for ease of integration in existing workflows
- Customizable labelling app design for varying user expertise
- Annotation support via [3DSlicer](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer)
  & [OHIF](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/ohif) for radiology
- Annotation support via [QuPath](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/qupath)
  , [Digital Slide Archive](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/dsa)
  & [CVAT](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/cvat) for
  pathology
- PACS connectivity via [DICOMWeb](https://www.dicomstandard.org/using/dicomweb)

## Installation

MONAI Label supports following OS with **GPU/CUDA** enabled.

- Ubuntu
- [Windows](https://docs.monai.io/projects/label/en/latest/installation.html#windows)

### [Current/Stable Version](https://pypi.org/project/monailabel/#history)

```bash
pip install monailabel -U
```


### Development version

To install the _**latest features**_ using one of the following options:

#### Git Checkout (developer mode)

```bash
git clone https://github.com/Project-MONAI/MONAILabel
pip install -r MONAILabel/requirements.txt
export PATH=$PATH:`pwd`/MONAILabel/monailabel/scripts
```

#### [Weekly Release](https://pypi.org/project/monailabel-weekly/)

```bash
pip install monailabel-weekly -U
```

#### [Docker](https://hub.docker.com/r/projectmonai/monailabel/tags)

```bash
docker run --gpus all --rm -ti --ipc=host --net=host projectmonai/monailabel:latest bash
```

---
Once the package is installed, you can download sample `radiology` or `pathology` app and start monailabel server.

```bash
# download radiology app and sample dataset
monailabel apps --download --name radiology --output apps
monailabel datasets --download --name Task09_Spleen --output datasets

# start server using radiology app with deepedit model enabled
monailabel start_server --app apps/radiology --studies datasets/Task09_Spleen/imagesTr --conf models deepedit
```

More details refer docs: https://docs.monai.io/projects/label/en/stable/installation.html

> If monailabel install path is not automatically determined, then you can provide explicit install path as:
> `monailabel apps --prefix ~/.local`

For **_prerequisites_**, other installation methods (using the default GitHub branch, using Docker, etc.), please refer
to the [installation guide](https://docs.monai.io/projects/label/en/latest/installation.html).

> Once you start the MONAI Label Server, by default server will be up and serving at http://127.0.0.1:8000/. Open the
> serving URL in browser. It will provide you the list of Rest APIs available. **For this, please make sure you use the
HTTP protocol.** _You can provide ssl arguments to run server in **HTTPS mode** but this functionality is not fully
verified across all clients._


### Optional Dependencies
Following are the optional dependencies which can help you to accelerate some GPU based transforms from MONAI.
These dependencies are by-default available if you are using `projectmonai/monailabel` docker.
- [CUCIM](https://pypi.org/project/cucim/)
- [CUPY](https://docs.cupy.dev/en/stable/install.html#installing-cupy)

## Plugins

### [3D Slicer](https://download.slicer.org/) (radiology)

Download **Preview Release** from https://download.slicer.org/ and install MONAI Label plugin from Slicer Extension
Manager.

Refer [3D Slicer plugin](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer) for other options to
install and run MONAI Label plugin in 3D Slicer.
> To avoid accidentally using an older Slicer version, you may want to _uninstall_ any previously installed 3D Slicer
> package.

### [OHIF](https://ohif.org/) (radiology)

MONAI Label comes with [pre-built plugin](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/ohif)
for [OHIF Viewer](https://github.com/OHIF/Viewers). To use OHIF
Viewer, you need to provide **DICOMWeb** instead of FileSystem as _studies_ when you start the server.
> Please install [Orthanc](https://www.orthanc-server.com/download.php) before using OHIF Viewer.
> For Ubuntu 20.x, Orthanc can be installed as `apt-get install orthanc orthanc-dicomweb`. However, you have to
> **upgrade to latest version** by following steps
> mentioned [here](https://book.orthanc-server.com/users/debian-packages.html#replacing-the-package-from-the-service-by-the-lsb-binaries).
>
> You can use [PlastiMatch](https://plastimatch.org/plastimatch.html#plastimatch-convert) to convert NIFTI to DICOM

```bash
  # start server using DICOMWeb
  monailabel start_server --app apps/radiology --studies http://127.0.0.1:8042/dicom-web
```

> OHIF Viewer will be accessible at http://127.0.0.1:8000/ohif/

![OHIF](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/ohif.png)

> **NOTE:** OHIF does not yet support Multi-Label interaction for DeepEdit.  And you can still use 3D Slicer when MONAILabel is connected to DICOMWeb.

### [QuPath](https://qupath.github.io/) (pathology)

You can download sample whole slide images
from [https://portal.gdc.cancer.gov/repository](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%5D%7D)

```bash
  # start server using pathology over downloaded whole slide images
  monailabel start_server --app apps/pathology --studies wsi_images
```

Refer [QuPath](plugins/qupath) for installing and running MONAILabel plugin in QuPath.

![image](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/qupath.jpg)

### [Digital Slide Archive (DSA)](https://digitalslidearchive.github.io/digital_slide_archive/) (pathology)

Refer [Pathology](sample-apps/pathology) for running a sample pathology use-case in MONAILabel.
> **NOTE:** The *DSA Plugin* is under *active development*.

![image](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/dsa.jpg)

### [CVAT](plugins/cvat)

Install [CVAT](https://openvinotoolkit.github.io/cvat/docs/getting_started) and
enable [Semi-Automatic and Automatic Annotation](https://openvinotoolkit.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/)
.
Refer [CVAT Instructions](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/cvat) for deploying available MONAILabel
pathology models into CVAT.

![image](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/cvat_detector.jpeg)

## Cite

If you are using MONAI Label in your research, please use the following citation:

  ```bash
@article{DiazPinto2022monailabel,
   author = {Diaz-Pinto, Andres and Alle, Sachidanand and Ihsani, Alvin and Asad, Muhammad and
            Nath, Vishwesh and P{\'e}rez-Garc{\'\i}a, Fernando and Mehta, Pritesh and
            Li, Wenqi and Roth, Holger R. and Vercauteren, Tom and Xu, Daguang and
            Dogra, Prerna and Ourselin, Sebastien and Feng, Andrew and Cardoso, M. Jorge},
    title = {{MONAI Label: A framework for AI-assisted Interactive Labeling of 3D Medical Images}},
  journal = {arXiv e-prints},
     year = 2022,
     url  = {https://arxiv.org/pdf/2203.12362.pdf}
}
  ```

## Contributing

For guidance on making a contribution to MONAI Label, see
the [contributing guidelines](https://github.com/Project-MONAI/MONAILabel/blob/main/CONTRIBUTING.md).

## Community

Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join
our [Slack channel](https://projectmonai.slack.com/archives/C031QRE0M1C).

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
