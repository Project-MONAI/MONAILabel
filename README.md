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

[MONAI Label Demo](https://youtu.be/o8HipCgSZIw?t=1319)

![DEMO](docs/images/demo.png)
<table>
<tr>
<td><img src="docs/images/ohif.png" alt="drawing" width="150"/></td>
<td><img src="docs/images/dsa.jpg" alt="drawing" width="150"/></td>
<td><img src="docs/images/qupath.jpg" alt="drawing" width="150"/></td>
<td><img src="docs/images/cvat_detector.jpeg" alt="drawing" width="150"/></td>
</tr>
</table>

## Features

> _The codebase is currently under active development._

- Framework for developing and deploying MONAI Label Apps to train and infer AI models
- Compositional & portable APIs for ease of integration in existing workflows
- Customizable labelling app design for varying user expertise
- Annotation support via [3DSlicer](plugins/slicer) & [OHIF](plugins/ohif) for radiology
- Annotation support via [QuPath](plugins/qupath) , [Digital Slide Archive](plugins/dsa) & [CVAT](plugins/cvat) for
  pathology
- PACS connectivity via DICOMWeb

## Installation

MONAI Label supports following OS with **GPU/CUDA** enabled.

- Ubuntu
- [Windows](https://docs.monai.io/projects/label/en/latest/installation.html#windows)

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
pip install monailabel-weekly
```

#### [Docker](https://hub.docker.com/r/projectmonai/monailabel/tags)

```bash
docker run --gpus all --rm -ti --ipc=host --net=host projectmonai/monailabel:latest
```

#### [Release Candidate](https://pypi.org/project/monailabel/#history)

```bash
pip install monailabel>=0.4*
```

Once the package is installed, you can download sample `radiology` or `pathology` app and start monailabel server.

```bash
# download radiology app and sample dataset
monailabel apps --download --name radiology --output apps
monailabel datasets --download --name Task09_Spleen --output datasets

# start server using radiology app with deepedit model enabled
monailabel start_server --app apps/radiology --studies datasets/Task09_Spleen/imagesTr --conf models deepedit
```

### Current/Stable version (0.3.x)

To install the [current release](https://pypi.org/project/monailabel/), you can simply run:

```bash
pip install monailabel

monailabel apps --download --name deepedit --output apps
monailabel datasets --download --name Task09_Spleen --output datasets

monailabel start_server --app apps/deepedit --studies datasets/Task09_Spleen/imagesTr
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

## Plugins

### [3D Slicer](https://download.slicer.org/) (radiology)

Download **Preview Release** from https://download.slicer.org/ and install MONAI Label plugin from Slicer Extension
Manager.

Refer [3D Slicer plugin](plugins/slicer) for other options to install and run MONAI Label plugin in 3D Slicer.
> To avoid accidentally using an older Slicer version, you may want to _uninstall_ any previously installed 3D Slicer
> package.

### [OHIF](https://ohif.org/) (radiology)

MONAI Label comes with [pre-built plugin](plugins/ohif) for [OHIF Viewer](https://github.com/OHIF/Viewers). To use OHIF
Viewer, you need to provide DICOMWeb instead of FileSystem as _studies_ when you start the server.
> Please install [Orthanc](https://www.orthanc-server.com/download.php) before using OHIF Viewer.
> For Ubuntu 20.x, Orthanc can be installed as `apt-get install orthanc orthanc-dicomweb`. However, you have to **
> upgrade to latest version** by following steps
> mentioned [here](https://book.orthanc-server.com/users/debian-packages.html#replacing-the-package-from-the-service-by-the-lsb-binaries)
>
> You can use [PlastiMatch](https://plastimatch.org/plastimatch.html#plastimatch-convert) to convert NIFTI to DICOM

```bash
  # start server using DICOMWeb
  monailabel start_server --app apps/radiology --studies http://127.0.0.1:8042/dicom-web
```

> OHIF Viewer will be accessible at http://127.0.0.1:8000/ohif/

![OHIF](docs/images/ohif.png)

> **_NOTE:_** OHIF does not yet support Multi-Label interaction for DeepEdit.

### [QuPath](https://qupath.github.io/) (pathology)

You can download sample whole slide images
from [https://portal.gdc.cancer.gov/repository](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%5D%7D)

```bash
  # start server using pathology over downloaded whole slide images
  monailabel start_server --app apps/pathology --studies wsi_images
```

Refer [QuPath](plugins/qupath) for installing and running MONAILabel plugin in QuPath.

![image](docs/images/qupath.jpg)

### [Digital Slide Archive (DSA)](https://digitalslidearchive.github.io/digital_slide_archive/) (pathology)

Refer [Pathology](sample-apps/pathology) for running a sample pathology use-case in MONAILabel.
> **_NOTE:_** The **Pathology App** and *DSA Plugin* are under *active development*.

![image](docs/images/dsa.jpg)

### [CVAT](plugins/cvat) (pathology)

Install [CVAT](https://openvinotoolkit.github.io/cvat/docs/getting_started) and
enable [Semi-Automatic and Automatic Annotation](https://openvinotoolkit.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/)
.
Refer [CVAT](plugins/cvat) for deploying available MONAILabel models into CVAT.

![image](docs/images/cvat_detector.jpeg)

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

For guidance on making a contribution to MONAI Label, see the [contributing guidelines](CONTRIBUTING.md).

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
