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

MONAI Label is an intelligent open source image labeling and learning tool that enables users to create annotated datasets and build AI annotation models for clinical evaluation. MONAI Label enables application developers to build labeling apps in a serverless way, where custom labeling apps are exposed as a service through the MONAI Label Server.

MONAI Label is a server-client system that facilitates interactive medical image annotation by using AI. It is an
open-source and easy-to-install ecosystem that can run locally on a machine with single or multiple GPUs. Both server
and client work on the same/different machine. It shares the same principles
with [MONAI](https://github.com/Project-MONAI).

Refer to full [MONAI Label documentations](https://docs.monai.io/projects/label/en/latest/index.html) for more details or check out our [MONAI Label Deep Dive videos series](https://www.youtube.com/playlist?list=PLtoSVSQ2XzyD4lc-lAacFBzOdv5Ou-9IA).



### Table of Contents
- [Overview](#Overview)
  - [Highlights and Features](#Highlights-and-Features)
  - [Supported Matrix](#Supported-Matrix)
- [Getting Started with MONAI Label](#Getting-Started-with-MONAI-Label)
  - [Step 1. Installation](#Step-1-Installation)
  - [Step 2. MONAI Label Sample Applications](#Step-2-MONAI-Label-Sample-Applications)
  - [Step 3. MONAI Label Supported Viewers](#Step-3-MONAI-Label-Supported-Viewers)
  - [Step 4. Data Preparation](#Step-4-Data-Preparation)
  - [Step 5. Start MONAI Label Server and Start Annotating!](#Step-5-Start-MONAI-Label-Server-and-Start-Annotating)
- [Cite MONAI Label](#Cite)
- [Contributing](#Contributing)
- [Community](#Community)
- [Additional Resources](#Additional-Resources)

### Overview
MONAI Label reduces the time and effort of annotating new datasets and enables the adaptation of AI to the task at hand by continuously learning from user interactions and data. MONAI Label allows researchers and developers to make continuous improvements to their apps by allowing them to interact with their apps at the user would. End-users (clinicians, technologists, and annotators in general) benefit from AI continuously learning and becoming better at understanding what the end-user is trying to annotate.

MONAI Label aims to fill the gap between developers creating new annotation applications, and the end users which want to benefit from these innovations.

#### Highlights and Features
- Framework for developing and deploying MONAI Label Apps to train and infer AI models
- Compositional & portable APIs for ease of integration in existing workflows
- Customizable labeling app design for varying user expertise
- Annotation support via [3DSlicer](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer)
  & [OHIF](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/ohif) for radiology
- Annotation support via [QuPath](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/qupath), [Digital Slide Archive](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/dsa), and [CVAT](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/cvat) for
  pathology
- Annotation support via [CVAT](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/cvat) for Endoscopy
- PACS connectivity via [DICOMWeb](https://www.dicomstandard.org/using/dicomweb)
- Automated Active Learning workflow for endoscopy using [CVAT](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/cvat)

#### Supported Matrix
Here you can find a table of the various supported fields, modalities, viewers, and general data types.  However, these are only ones that we've explicitly test and that doesn't mean that your dataset or file type won't work with MONAI Label.  Try MONAI for your given task and if you're having issues, reach out through GitHub Issues.
<table>
<tr>
  <th>Field</th>
  <th>Models</th>
  <th>Viewers</th>
  <th>Data Types</th>
  <th>Image Modalities/Target</th>
</tr>
  <td>Radiology</td>
  <td>
    <ul>
      <li>Segmentation</li>
      <li>DeepGrow</li>
      <li>DeepEdit</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>3DSlicer</li>
      <li>OHIF</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>NIfTI</li>
      <li>NRRD</li>
      <li>DICOM</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>CT</li>
      <li>MRI</li>
    </ul>
  </td>
<tr>
</tr>
  <td>Pathology</td>
  <td>
    <ul>
      <li>DeepEdit</li>
      <li>NuClick</li>
      <li>Segmentation</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>Digital Slide Archive</li>
      <li>QuPath</li>
      <li>CVAT</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>TIFF</li>
      <li>SVS</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>Nuclei Segmentation</li>
      <li>Nuclei Classification</li>
    </ul>
  </td>
<tr>
</tr>
  <td>Video</td>
  <td>
    <ul>
      <li>DeepEdit</li>
      <li>Tooltracking</li>
      <li>InBody/OutBody</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>CVAT</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>JPG</li>
      <li>3-channel Video Frames</li>
    </ul>
  </td>
  <td>
    <ul>
      <li>Endoscopy</li>
    </ul>
  </td>
<tr>
</table>

# Getting Started with MONAI Label
### MONAI Label requires a few steps to get started:
- Step 1: [Install MONAI Label](#Step-1-Installation)
- Step 2: [Download a MONAI Label sample app or write your own custom app](#Step-2-MONAI-Label-Sample-Applications)
- Step 3: [Install a compatible viewer and supported MONAI Label Plugin](#Step-3-MONAI-Label-Supported-Viewers)
- Step 4: [Prepare your Data](#Step-4-Data-Preparation)
- Step 5: [Launch MONAI Label Server and start Annotating!](#Step-5-Start-MONAI-Label-Server-and-Start-Annotating)

## Step 1 Installation

### Current Stable Version
<a href="https://pypi.org/project/monailabel/#history"><img alt="GitHub release (latest SemVer)" src="https://img.shields.io/github/v/release/project-monai/monailabel"></a>
<pre>pip install -U monailabel</pre>

MONAI Label supports the following OS with **GPU/CUDA** enabled. For more details instruction, please see the installation guides.
- [Ubuntu](https://docs.monai.io/projects/label/en/latest/installation.html)
- [Windows](https://docs.monai.io/projects/label/en/latest/installation.html#windows)

### GPU Acceleration (Optional Dependencies)
Following are the optional dependencies which can help you to accelerate some GPU based transforms from MONAI. These dependencies are enabled by default if you are using `projectmonai/monailabel` docker.
- [CUCIM](https://pypi.org/project/cucim/)
- [CUPY](https://docs.cupy.dev/en/stable/install.html#installing-cupy)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Development version

To install the _**latest features**_ using one of the following options:

<details>
  <summary><strong>Git Checkout (developer mode)</strong></summary>
  <a href="https://github.com/Project-MONAI/MONAILabel"><img alt="GitHub tag (latest SemVer)" src="https://img.shields.io/github/v/tag/Project-MONAI/monailabel"></a>
  <br>
  <pre>
  git clone https://github.com/Project-MONAI/MONAILabel
  pip install -r MONAILabel/requirements.txt
  export PATH=$PATH:`pwd`/MONAILabel/monailabel/scripts</pre>
  <p>If you are using DICOM-Web + OHIF then you have to build OHIF package separate.  Please refer [here](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/ohif#development-setup).</p>
</details>

<details>
  <summary><strong>Weekly Release</strong></summary>
  <a href="https://badge.fury.io/py/monailabel-weekly"><img src="https://badge.fury.io/py/monailabel-weekly.svg" alt="PyPI version" height="18"></a>
  <br>
  <pre>pip install monailabel-weekly -U</pre>
</details>

<details>
  <summary><strong>Docker</strong></summary>
  <img alt="Docker Image Version (latest semver)" src="https://img.shields.io/docker/v/projectmonai/monailabel">
  <br>
  <pre>docker run --gpus all --rm -ti --ipc=host --net=host projectmonai/monailabel:latest bash</pre>
</details>

## Step 2 MONAI Label Sample Applications

<h3>Radiology</h3>
<p>This app has example models to do both interactive and automated segmentation over radiology (3D) images. Including auto segmentation with the latest deep learning models (e.g., UNet, UNETR) for multiple abdominal organs. Interactive tools include DeepEdit and Deepgrow for actively improving trained models and deployment.</p>
<ul>
  <li>Deepedit</li>
  <li>Deepgrow</li>
  <li>Segmentation</li>
  <li>Spleen Segmentation</li>
  <li>Multi-Stage Vertebra Segmentation</li>
</ul>

<h3>Pathology</h3>
<p>This app has example models to do both interactive and automated segmentation over pathology (WSI) images. Including nuclei multi-label segmentation for Neoplastic cells, Inflammatory, Connective/Soft tissue cells, Dead Cells, and Epithelial. The app provides interactive tools including DeepEdits for interactive nuclei segmentation.</p>
<ul>
  <li>Deepedit</li>
  <li>Deepgrow</li>
  <li>Segmentation</li>
  <li>Spleen Segmentation</li>
  <li>Multi-Stage Vertebra Segmentation</li>
</ul>
<h3>Video</h3>
<p>The Endoscopy app enables users to use interactive, automated segmentation and classification models over 2D images for endoscopy usecase. Combined with CVAT, it will demonstrate the fully automated Active Learning workflow to train + fine-tune a model.</p>
<ul>
  <li>Deepedit</li>
  <li>ToolTracking</li>
  <li>InBody/OutBody</li>
</ul>
<h3>Bundles</h3>
<p>The Bundle app enables users with customized models for inference, training or pre and post processing any target anatomies. The specification for MONAILabel integration of the Bundle app links archived Model-Zoo for customized labeling (e.g., the third-party transformer model for labeling renal cortex, medulla, and pelvicalyceal system. Interactive tools such as DeepEdits).</p>

For a full list of supported bundles, see the <a href="https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/monaibundle">MONAI Label Bundles README</a>.

## Step 3 MONAI Label Supported Viewers

### Radiology
#### 3D Slicer
3D Slicer, a free and open-source platform for analyzing, visualizing and understanding medical image data. In MONAI Label, 3D Slicer is most tested with radiology studies and algorithms, develpoment and integration.

[3D Slicer Setup]()

#### OHIF
The Open Health Imaging Foundation (OHIF) Viewer is an open source, web-based, medical imaging platform. It aims to provide a core framework for building complex imaging applications.

[OHIF Setup]()

### Pathology
#### QuPath
Quantitative Pathology & Bioimage Analysis (QuPath) is an open, powerful, flexible, extensible software platform for bioimage analysis.

[QuPath Setup]()

#### Digital Slide Archive
The Digital Slide Archive (DSA) is a platform that provides the ability to store, manage, visualize and annotate large imaging data sets.
[Digital Slide Archive Setup]()

### Video
#### CVAT
CVAT is an interactive video and image annotation tool for computer vision.
[CVAT Setup]()

## Step 4 Data Preparation
For data preparation, you have two options, you can use a local data store or any image archive tool that supports DICOMWeb.

#### Local Datastore for the Radiology App on single modality images
For a Datastore in a local file archive, there is a set folder structure that MONAI Label uses. Place your image data in a folder and if you have any segmentation files, create and place them in a subfolder called `labels/final`. You can see an example below:
```
dataset
│-- spleen_10.nii.gz
│-- spleen_11.nii.gz
│   ...
└───labels
    └─── final
        │-- spleen_10.nii.gz
        │-- spleen_11.nii.gz
        │   ...
```

If you don't have labels, just place the images/volumes in the dataset folder.

#### DICOMWeb Support
If the viewer you're using supports DICOMweb standard, you can use that instead of a local datastore to serve images to MONAI Label. When starting the MONAI Label server, we need to specify the URL of the DICOMweb service in the studies argument (and, optionally, the username and password for DICOM servers that require them). You can see an example of starting the MONAI Label server with a DICOMweb URL below:


```
monailabel start_server --app apps/radiology --studies http://127.0.0.1:8042/dicom-web --conf models segmentation
```


## Step 5 Start MONAI Label Server and Start Annotating
You're now ready to start using MONAI Label.  Once you've configured your viewer, app, and datastore, you can launch the MONAI Label server with the relevant parameters. For simplicity, you can see an example where we download a Radiology sample app and dataset, then start the MONAI Label server below:

```
monailabel apps --download --name radiology --output apps
monailabel datasets --download --name Task09_Spleen --output datasets
monailabel start_server --app apps/radiology --studies datasets/Task09_Spleen/imagesTr --conf models segmentation
```

**Note:** If you want to work on different labels than the ones proposed by default, change the configs file following the instructions here: https://youtu.be/KtPE8m0LvcQ?t=622

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

@inproceedings{DiazPinto2022DeepEdit,
      title={{DeepEdit: Deep Editable Learning for Interactive Segmentation of 3D Medical Images}},
      author={Diaz-Pinto, Andres and Mehta, Pritesh and Alle, Sachidanand and Asad, Muhammad and Brown, Richard and Nath, Vishwesh and Ihsani, Alvin and Antonelli, Michela and Palkovics, Daniel and Pinter, Csaba and others},
      booktitle={MICCAI Workshop on Data Augmentation, Labelling, and Imperfections},
      pages={11--21},
      year={2022},
      organization={Springer}
}
 ```

Optional Citation: if you are using active learning functionality from MONAI Label, please support us:

```bash
@article{nath2020diminishing,
  title={Diminishing uncertainty within the training pool: Active learning for medical image segmentation},
  author={Nath, Vishwesh and Yang, Dong and Landman, Bennett A and Xu, Daguang and Roth, Holger R},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={10},
  pages={2534--2547},
  year={2020},
  publisher={IEEE}
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

## Additional Resources

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
- Client API: https://www.youtube.com/watch?v=mPMYJyzSmyo
- Demo Videos: https://www.youtube.com/c/ProjectMONAI
