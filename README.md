# MONAILabel

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/MONAILabel/workflows/build/badge.svg?branch=main)](https://github.com/Project-MONAI/MONAILabel/commits/main)
[![Documentation Status](https://readthedocs.org/projects/monailabel/badge/?version=latest)](https://docs.monai.io/projects/label/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAILabel/branch/main/graph/badge.svg)](https://codecov.io/gh/Project-MONAI/MONAILabel)
[![PyPI version](https://badge.fury.io/py/monailabel.svg)](https://badge.fury.io/py/monailabel-weekly)

MONAILabel is a server-client system that facilitates interactive medical image annotation by using AI. It is an
open-source and easy-to-install ecosystem that can run locally on a machine with one or two GPUs. Both server and client
work on the same/different machine. However, initial support for multiple users is restricted. It shares the same
principles with [MONAI](https://github.com/Project-MONAI).

[Brief Demo](https://www.youtube.com/watch?v=vFirnscuOVI)

<img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/demo.png" width="800"/>

> **Development in Progress**.
> We will be actively working on this repository to add more features, fix issues, update docs, readme etc...
> as we make more progress. Wiki's, LICENSE, Contributions, Code Compliance, CI Tool Integration etc... Otherwise, it shares the same
principles with [MONAI](https://github.com/Project-MONAI).

## Features
> _The codebase is currently under active development._

- framework for developing and deploying MONAILabel Apps to train and infer AI models
- compositional & portable APIs for ease of integration in existing workflows
- customizable design for varying user expertise
- 3D slicer support


## Installation
MONAILabel supports following OS with GPU/CUDA enabled.

### Ubuntu
```bash
  # One time setup (to pull monai with nvidia gpus)
  docker run -it --rm --gpus all --ipc=host --net=host -v /rapid/xyz:/workspace/ projectmonai/monai:0.5.2
  
  # Install monailabel 
  pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel
  
  # Download MSD Datasets
  monailabel datasets # list sample datasets
  monailabel datasets --download --name Task02_Heart --output /workspace/datasets/
  
  # Download Sample Apps
  monailabel apps # list sample apps
  monailabel apps --download --name deepedit_left_atrium --output /workspace/apps/
  
  # Start Server
  monailabel start_server --app /workspace/apps/deepedit_left_atrium --studies /workspace/datasets/Task02_Heart/imagesTr
```

### Windows

#### Pre Requirements
Make sure you have python 3.x version environment with PyTorch + CUDA installed.
- Install [python](https://www.python.org/downloads/)
- Install [cuda](https://developer.nvidia.com/cuda-downloads) (Faster mode: install CUDA runtime only)
- `python -m pip install --upgrade pip setuptools wheel`
- `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
- `python -c "import torch; print(torch.cuda.is_available())"`

#### MONAILabel

```bash
  pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel
  monailabel -h
  
  # Download MSD Datasets
  monailabel datasets # List sample datasets
  monailabel datasets --download --name Task02_Heart --output C:\Workspace\Datasets
  
  # Download Sample Apps
  monailabel apps # List sample apps
  monailabel apps --download --name deepedit_left_atrium --output C:\Workspace\Apps
  
  # Start Server
  monailabel start_server --app C:\Workspace\Apps\deepedit_left_atrium --studies C:\Workspace\Datasets\Task02_Heart\imagesTr
```

> Once you start the MONAILabel Server, by default it will be up and serving at http://127.0.0.1:8000/. Open the serving
  URL in browser. It will provide you the list of Rest APIs available.

### 3D Slicer

Refer [3D Slicer plugin](plugins/slicer) for installing and running MONAILabel plugin in 3D Slicer.

## Contributing
For guidance on making a contribution to MONAILabel, see the [contributing guidelines](CONTRIBUTING.md).

## Community
Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join our [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

Ask and answer questions over on [MONAILabel's GitHub Discussions tab](https://github.com/Project-MONAI/MONAILabel/discussions).

## Links
- Website: https://monai.io/
- API documentation: https://docs.monai.io/monailabel
- Code: https://github.com/Project-MONAI/MONAILabel
- Project tracker: https://github.com/Project-MONAI/MONAILabel/projects
- Issue tracker: https://github.com/Project-MONAI/MONAILabel/issues
- Wiki: https://github.com/Project-MONAI/MONAILabel/wiki
- Test status: https://github.com/Project-MONAI/MONAILabel/actions
- PyPI package: https://pypi.org/project/monailabel/
- Weekly previews: https://pypi.org/project/monailabel-weekly/
- Docker Hub: https://hub.docker.com/r/projectmonai/monailabel
