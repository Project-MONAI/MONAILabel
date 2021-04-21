# MONAI-label

Draft Repository for MONAI Label

## UseCase

![alt text](https://www.websequencediagrams.com/cgi-bin/cdraw?lz=dGl0bGUgU2ltcGxlIFVzZWNhc2UKClJlc2VhcmNoZXItPgACCjogcGlwIGluc3RhbGwgbW9uYWlsYWJlbAAWGWRldmVsb3BzIG15X2FwcApub3RlIG92ZXIgAEkMdXNpbmcgdGVtcGxhdGUgYXBwIHByb3ZpZGVzIFxuMS4gdHJhaW5cbjIuIGluZmVyIGZvciBwcmVfABMFZWQgbW9kZWxzXG4zLiBhY3RpdmUgbGVhcm4gc3RyYXRlZ2llcwCBRA1NT05BSUxhYmVsOiBzdGFydF8AgUYFXwCBRwUoAIElBiwgZGF0YXNldCkAgX4OYWRpb2xvZ2lzdDogc2hhcmUAggEGIACCAwUgc2VydmVyIGlwCgAeCy0-M0RTbGljZXI6IGNvbmZpZ3VyZQCBfwwASAxGZXcAHgdzIGFyZSBhdgCCXwVsZSB0byBzZWxlY3QgXG5uZXR3b3JrLwCBWgd5L2RldmljZSBldGMuLgoAZwgAgVwObmV4dF9zYW1wbGUoAIIgBkxlYXJuaW5nKQoAgggKAIEhDAAnBiBkZXRhaWxzAEwLAIFGCmZldGNoAE4IbG9jYWwvcmVtb3RlKQBvF3J1bgCDPwZlbmNlKGRlZXBncm93LCBhdXRvc2VnbWVudAB3GACEZAcAgkEYcnJlY3RzAB8HAIFtFnN1Ym1pdCgAhS8FAIFnDgCEBwwAhGkFKG5ldwCCJwcpCgo&s=rose)

## Installation

```bash
# One time setup (docker to pull nvidia gpus and pytorch)
docker run -it --rm --gpus all --ipc=host --net=host -v /rapid/xyz:/workspace/ nvcr.io/nvidia/pytorch:21.02-py3 
git clone git@github.com:SachidanandAlle/MONAI-label.git /workspace/MONAI-label
cd /workspace/MONAI-label

pip install -r requirements.txt

# Download sample pre-trained models
# https://drive.google.com/drive/folders/1QNEtb3InzpEve53xTWCmxe_HnR2cImUv?usp=sharing
cd /workspace/MONAI-label/apps/my_app
unzip models.zip

# Download MSD Datasets from
# https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
mkdir -p /workspace/datasets
cd /workspace/datasets
tar xf MSD_Task09_Spleen.tar

# Run APP
cd /workspace/MONAI-label/monailabel
export PYTHONPATH=/workspace/MONAI-label

python main.py --app ../apps/my_app/ --studies /workspace/datasets/MSD_Task09_Spleen

# To run in virtual env for the app by installing my_app/requirements.txt
#./start_monai_label.sh --app ../apps/my_app/ --studies
```

## App basic structure

<img src="https://user-images.githubusercontent.com/7339051/114428020-98cdaf80-9bb3-11eb-8010-40f47d1afcd6.png" width="200"/>

## Dry Run basic flow (without slicer)

- get /info/ (you get list of pre-trained models available) and other configs you can see for client
- post /activelearning/sample (you get next sample/image details)
- post /inference/segmentation_spleen (image is the one which u got in step 2)
- post /activelearning/label (not complete implemented yet.. but it will eventually save the label)
- post /train (manually start training.. otherwise plan is to add a config to support auto-start after N samples saved)
- If slicer is on remote it will use /download/ api to get it from server (u get the download link in sample details)

<img src="https://user-images.githubusercontent.com/7339051/115477603-31ab9d00-a23c-11eb-85f0-0b8ac374a9a0.png" width="500"/>


## Slicer

Install Plugin in developer mode

- git clone git@github.com:SachidanandAlle/MONAI-label.git
- Open 3D Slicer: Go to Edit -> Application Settings -> Modules -> Additional Module Paths
- Add New Module Path: <FULL_PATH>/clients/slicer/MONAILabel
- Restart

<img src="https://user-images.githubusercontent.com/7339051/115478017-1725f380-a23d-11eb-9b60-19638187b8e6.png" width="400"/>

