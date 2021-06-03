# MONAILabel

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/MONAILabel/workflows/build/badge.svg?branch=main)](https://github.com/Project-MONAI/MONAILabel/commits/main)
[![Documentation Status](https://readthedocs.org/projects/monai/badge/?version=latest)](https://docs.monai.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAILabel/branch/main/graph/badge.svg)](https://codecov.io/gh/Project-MONAI/MONAILabel)
[![PyPI version](https://badge.fury.io/py/monailabel.svg)](https://badge.fury.io/py/monailabel)

The MONAI-label is a server-client system that facilitates interactive medical image annotation by using AI. It is an
open-source and easy-to-install ecosystem that can run locally on a machine with one or two GPUs. Both server and client
work on the same/different machine. However, initial support for multiple users is restricted. It shares the same
principles with [MONAI](https://github.com/Project-MONAI).

[Brief Demo](https://www.youtube.com/watch?v=vFirnscuOVI)

<img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/demo.png" width="800"/>

> **Development in Progress**.
> We will be actively working on this repository to add more features, fix issues, update docs, readme etc...
> as we make more progress. Wiki's, LICENSE, Contributions, Code Compliance, CI Tool Integration etc... This will follow similar to [MONAI repository](https://github.com/Project-MONAI).

## Installation

- Pre-Trained models are available
  at [dropbox](https://www.dropbox.com/sh/gcobuwui5v2r8f5/AAAaJ3uFajwo4NRnQ0BqU46Ma?dl=0)
- Sample images/datasets can be downloaded
  from [monai-aws](https://github.com/Project-MONAI/MONAI/blob/master/monai/apps/datasets.py#L213-L224)

### Ubuntu

```bash
    # One time setup (to pull monai with nvidia gpus)
    docker run -it --rm --gpus all --ipc=host --net=host -v /rapid/xyz:/workspace/ projectmonai/monai:0.5.2

    # Install monailabel 
    pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel

    # Download MSD Datasets
    monailabel datasets
    monailabel datasets --download --name Task02_Heart --output /workspace/datasets/
    
    # Download Sample Apps
    monailabel apps
    monailabel apps --download --name deepedit_heart --output /workspace/apps/
    
    # Run APP
    monailabel run --app /workspace/apps/deepedit_heart --studies /workspace/datasets/Task02_Heart/imagesTr
```

### Windows

#### Pre Requirements

- Install [python](https://www.python.org/downloads/)
- Install [cuda](https://developer.nvidia.com/cuda-downloads) (Faster mode: install CUDA runtime only)
- `python -m pip install --upgrade pip setuptools wheel`
- `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

#### MONAILabel

```bash
    pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel
    monailabel -h

    # Download MSD Datasets
    monailabel datasets
    monailabel datasets --download --name Task02_Heart --output C:\Workspace\Datasets

    # Download Sample Apps
    monailabel apps
    monailabel apps --download --name deepedit_heart --output C:\Workspace\Apps

    # Run App
    monailabel run --app C:\Workspace\Apps\deepedit_heart --studies C:\Workspace\Datasets\Task02_Heart\imagesTr
```

## App basic structure

<img src="https://user-images.githubusercontent.com/7339051/120267190-61b67900-c29b-11eb-8eaf-9c2bfa74f837.png" width="200"/>

## REST APIs

- Once you start the MONAILabel Server, by default it will be up and serving at http://127.0.0.1:8000/. Open the serving
  URL in browser. It will provide you the list of Rest APIs. You can try them with the need of actual viewer to dry run
  the features of MONAILabel.

<img src="https://user-images.githubusercontent.com/7339051/120266924-cd4c1680-c29a-11eb-884e-a60975981df9.png" width="500"/>

## 3D Slicer

Refer [3D Slicer plugin](plugins/slicer) for installing and running MONAILabel plugin in 3D Slicer.
