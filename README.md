# MONAILabel

The MONAI-label is a server-client system that facilitates interactive medical image annotation by using AI.
It is an open-source and easy-to-install ecosystem that can run locally on a machine with one or two GPUs.
Both server and client work on the same/different machine.  However, initial support for multiple users is restricted.
It shares the same principles with [MONAI](https://github.com/Project-MONAI).

[![Brief Demo]("docs/images/demo.png")](https://emckclac-my.sharepoint.com/:v:/g/personal/k2039747_kcl_ac_uk/EVlFnsuWNvFLuc2Nx37r2_YB1Kz9u8oOoo7o1GfiFa0_nQ?e=ANofe0)

<img src="docs/images/demo.png" width="800"/>

> **Development in Progress**. 
> We will be actively working on this repository to add more features, fix issues, update docs, readme etc... 
> as we make more progress.  Wiki's, LICENSE, Contributions, Code Compliance, CI Tool Integration etc... will follow similar to [MONAI repository](https://github.com/Project-MONAI).


## Simple UseCase

![alt text](https://www.websequencediagrams.com/cgi-bin/cdraw?lz=dGl0bGUgU2ltcGxlIFVzZWNhc2UKClJlc2VhcmNoZXItPgACCjogcGlwIGluc3RhbGwgbW9uYWlsYWJlbAAWGWRldmVsb3BzIG15X2FwcApub3RlIG92ZXIgAEkMdXNpbmcgdGVtcGxhdGUgYXBwIHByb3ZpZGVzIFxuMS4gdHJhaW5cbjIuIGluZmVyIGZvciBwcmVfABMFZWQgbW9kZWxzXG4zLiBhY3RpdmUgbGVhcm4gc3RyYXRlZ2llcwCBRA1NT05BSUxhYmVsOiBzdGFydF8AgUYFXwCBRwUoAIElBiwgZGF0YXNldCkAgX4OYWRpb2xvZ2lzdDogc2hhcmUAggEGIACCAwUgc2VydmVyIGlwCgAeCy0-M0RTbGljZXI6IGNvbmZpZ3VyZQCBfwwASAxGZXcAHgdzIGFyZSBhdgCCXwVsZSB0byBzZWxlY3QgXG5uZXR3b3JrLwCBWgd5L2RldmljZSBldGMuLgoAZwgAgVwObmV4dF9zYW1wbGUoAIIgBkxlYXJuaW5nKQoAgggKAIEhDAAnBiBkZXRhaWxzAEwLAIFGCmZldGNoAE4IbG9jYWwvcmVtb3RlKQBvF3J1bgCDPwZlbmNlKGRlZXBncm93LCBhdXRvc2VnbWVudAB3GACEZAcAgkEYcnJlY3RzAB8HAIFtFnN1Ym1pdCgAhS8FAIFnDgCEBwwAhGkFKG5ldwCCJwcpCgo&s=rose)

## Installation *(Development Mode)*
 - Pre-Trained models are available at [dropbox](https://www.dropbox.com/sh/gcobuwui5v2r8f5/AAAaJ3uFajwo4NRnQ0BqU46Ma?dl=0)
 - Download sample images/datasets from [monai-aws](https://github.com/Project-MONAI/MONAI/blob/master/monai/apps/datasets.py#L213-L224)

### Ubuntu
```bash
# One time setup (to pull monai with nvidia gpus)
docker run -it --rm --gpus all --ipc=host --net=host -v /rapid/xyz:/workspace/ projectmonai/monai:0.5.2
git clone git@github.com:Project-MONAI/MONAILabel.git /workspace/MONAILabel
cd /workspace/MONAILabel

pip install -r requirements.txt

# Download MSD Datasets from
mkdir -p /workspace/datasets
cd /workspace/datasets
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
tar xf Task09_Spleen.tar

# Run APP
cd /workspace/MONAILabel/monailabel
export PYTHONPATH=/workspace/MONAILabel

python main.py --app ../sample-apps/deepgrow/ --studies /workspace/datasets/Task09_Spleen/imagesTr
```

### Windows
 1. Install python and pip
    >Install python from https://www.python.org/downloads/

    `python -m pip install --upgrade pip setuptools wheel`

 2. Install cuda (Faster mode: install CUDA runtime only)
    >https://developer.nvidia.com/cuda-downloads


 3. Install torch (https://pytorch.org/get-started/locally/)

    `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`


 4. Setup MONAILabel

    ```bash
    git clone git@github.com:Project-MONAI/MONAILabel.git C:/Projects
    pip install -r C:/Projects/MONAILabel/requirements.txt

    # Download sample datasets
    ```


 5. Run App
    ```bash
    set PYTHONPATH=C:/Projects/MONAILabel
    cd C:/Projects/MONAILabel

    python monailabel\main.py -a sample-apps\deepedit_heart -s C:\Datasets\Task02_Heart\imagesTr
    ```


## App basic structure

<img src="https://user-images.githubusercontent.com/7339051/114428020-98cdaf80-9bb3-11eb-8010-40f47d1afcd6.png" width="200"/>

## Dry Run basic flow (without slicer)

- http://127.0.0.1:8000/ will provide you the list of Rest APIs and click them to try any of them.

<img src="https://user-images.githubusercontent.com/7339051/115477603-31ab9d00-a23c-11eb-85f0-0b8ac374a9a0.png" width="500"/>


## Slicer

Install Plugin in developer mode

- git clone git@github.com:Project-MONAI/MONAILabel.git
- Open 3D Slicer: Go to Edit -> Application Settings -> Modules -> Additional Module Paths
- Add New Module Path: <FULL_PATH>/plugins/slicer/MONAILabel
- Restart

<img src="https://user-images.githubusercontent.com/7339051/115478017-1725f380-a23d-11eb-9b60-19638187b8e6.png" width="400"/>
