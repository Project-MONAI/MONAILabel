# MONAILabel

The MONAI-label is a server-client system that facilitates interactive medical image annotation by using AI.
It is an open-source and easy-to-install ecosystem that can run locally on a machine with one or two GPUs.
Both server and client work on the same/different machine.  However, initial support for multiple users is restricted.
It shares the same principles with [MONAI](https://github.com/Project-MONAI).


> **Development in Progress**. 
> We will be actively working on this repository to add more features, fix issues, update docs, readme etc... 
> as we make more progress.  Wiki's, LICENSE, Contributions, Code Compliance, CI Tool Integration etc... will follow similar to [MONAI repository](https://github.com/Project-MONAI).


## Simple UseCase

![alt text](https://www.websequencediagrams.com/cgi-bin/cdraw?lz=dGl0bGUgU2ltcGxlIFVzZWNhc2UKClJlc2VhcmNoZXItPgACCjogcGlwIGluc3RhbGwgbW9uYWlsYWJlbAAWGWRldmVsb3BzIG15X2FwcApub3RlIG92ZXIgAEkMdXNpbmcgdGVtcGxhdGUgYXBwIHByb3ZpZGVzIFxuMS4gdHJhaW5cbjIuIGluZmVyIGZvciBwcmVfABMFZWQgbW9kZWxzXG4zLiBhY3RpdmUgbGVhcm4gc3RyYXRlZ2llcwCBRA1NT05BSUxhYmVsOiBzdGFydF8AgUYFXwCBRwUoAIElBiwgZGF0YXNldCkAgX4OYWRpb2xvZ2lzdDogc2hhcmUAggEGIACCAwUgc2VydmVyIGlwCgAeCy0-M0RTbGljZXI6IGNvbmZpZ3VyZQCBfwwASAxGZXcAHgdzIGFyZSBhdgCCXwVsZSB0byBzZWxlY3QgXG5uZXR3b3JrLwCBWgd5L2RldmljZSBldGMuLgoAZwgAgVwObmV4dF9zYW1wbGUoAIIgBkxlYXJuaW5nKQoAgggKAIEhDAAnBiBkZXRhaWxzAEwLAIFGCmZldGNoAE4IbG9jYWwvcmVtb3RlKQBvF3J1bgCDPwZlbmNlKGRlZXBncm93LCBhdXRvc2VnbWVudAB3GACEZAcAgkEYcnJlY3RzAB8HAIFtFnN1Ym1pdCgAhS8FAIFnDgCEBwwAhGkFKG5ldwCCJwcpCgo&s=rose)

## Installation *(Development Mode)*
 - Pre-Trained models are available at [dropbox](https://www.dropbox.com/sh/gcobuwui5v2r8f5/AAAaJ3uFajwo4NRnQ0BqU46Ma?dl=0)
 - Download sample images/datasets from [monai-aws](https://github.com/Project-MONAI/MONAI/blob/master/monai/apps/datasets.py#L213-L224)

```bash
# One time setup (docker to pull nvidia gpus and pytorch)
docker run -it --rm --gpus all --ipc=host --net=host -v /rapid/xyz:/workspace/ nvcr.io/nvidia/pytorch:21.02-py3
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

## App basic structure

<img src="https://user-images.githubusercontent.com/7339051/114428020-98cdaf80-9bb3-11eb-8010-40f47d1afcd6.png" width="200"/>

## Dry Run basic flow (without slicer)

- get /info/ (you get list of pre-trained models available) and other configs you can see for client
- post /activelearning/sample (you get next sample/image details based on strategy)
- post /inference/segmentation_spleen (image is the one which you got in step 2)
- post /activelearning/label (this will save the label)
- post /train (manually start training.. otherwise plan is to add a config to support auto-start after N samples saved)
- If slicer is on remote it will use /download/ api to get it from server (you get the download link in sample details)

<img src="https://user-images.githubusercontent.com/7339051/115477603-31ab9d00-a23c-11eb-85f0-0b8ac374a9a0.png" width="500"/>


## Slicer

Install Plugin in developer mode

- git clone git@github.com:Project-MONAI/MONAILabel.git
- Open 3D Slicer: Go to Edit -> Application Settings -> Modules -> Additional Module Paths
- Add New Module Path: <FULL_PATH>/plugins/slicer/MONAILabel
- Restart

<img src="https://user-images.githubusercontent.com/7339051/115478017-1725f380-a23d-11eb-9b60-19638187b8e6.png" width="400"/>
