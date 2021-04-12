# MONAI-label

Draft Repository for MONAI Label

## UseCase

![alt text](https://www.websequencediagrams.com/cgi-bin/cdraw?lz=dGl0bGUgU2ltcGxlIFVzZWNhc2UKClJlc2VhcmNoZXItPgACCjogcGlwIGluc3RhbGwgbW9uYWlsYWJlbAAWGWRldmVsb3BzIG15X2FwcApub3RlIG92ZXIgAEkMdXNpbmcgdGVtcGxhdGUgYXBwIHByb3ZpZGVzIFxuMS4gdHJhaW5cbjIuIGluZmVyIGZvciBwcmVfABMFZWQgbW9kZWxzXG4zLiBhY3RpdmUgbGVhcm4gc3RyYXRlZ2llcwCBRA1NT05BSUxhYmVsOiBzdGFydF8AgUYFXwCBRwUoAIElBiwgZGF0YXNldCkAgX4OYWRpb2xvZ2lzdDogc2hhcmUAggEGIACCAwUgc2VydmVyIGlwCgAeCy0-M0RTbGljZXI6IGNvbmZpZ3VyZQCBfwwASAxGZXcAHgdzIGFyZSBhdgCCXwVsZSB0byBzZWxlY3QgXG5uZXR3b3JrLwCBWgd5L2RldmljZSBldGMuLgoAZwgAgVwObmV4dF9zYW1wbGUoAIIgBkxlYXJuaW5nKQoAgggKAIEhDAAnBiBkZXRhaWxzAEwLAIFGCmZldGNoAE4IbG9jYWwvcmVtb3RlKQBvF3J1bgCDPwZlbmNlKGRlZXBncm93LCBhdXRvc2VnbWVudAB3GACEZAcAgkEYcnJlY3RzAB8HAIFtFnN1Ym1pdCgAhS8FAIFnDgCEBwwAhGkFKG5ldwCCJwcpCgo&s=rose)

## Installation

```bash
# One time setup (docker to pull nvidia gpus and pytorch)
docker run -it --rm --gpus all --ipc=host --net=host -v /rapid/xyz:/workspace/ nvcr.io/nvidia/pytorch:21.02-py3 
git clone clone git@github.com:SachidanandAlle/MONAI-label.git /workspace/MONAI-label
cd /workspace/MONAI-label

pip install -r requirements.txt

# Download sample pre-trained models and studies from
# https://drive.google.com/drive/folders/1QNEtb3InzpEve53xTWCmxe_HnR2cImUv?usp=sharing
cd /workspace/MONAI-label/apps/my_app && unzip models.zip
cd /workspace/MONAI-label/apps/datasets && unzip studies.zip

# Run APP
cd /workspace/MONAI-label/monailabel
export PYTHONPATH=/workspace/dlmed/MONAI-label

# To run in virtual env for the app by installing my_app/requirements.txt
./start_monai_label.sh --app ../apps/my_app/ --studies ../datasets/studies 

# If you want to run in current env
# python main.py --app ../apps/my_app --studies ../datasets/studies

```

## App basic structure

<img src="https://user-images.githubusercontent.com/7339051/114417165-438ca080-9ba9-11eb-8642-483a62e51e4d.png" width="200"/>

## Dry Run basic flow (without slicer)

- get /info/ (you get list of pre-trained models available) and other configs you can see for client
- post /activelearning/next_sample (you get sample image details)
- post /inference/segmentation_spleen (image is the one which u got in step 2)
- post /activelearning/save_label (not complete implemented yet.. but it will eventually save the label)
- post /train (manually start training.. otherwise plan is to add a config to support auto-start after N samples saved)
- If slicer is on remote it will use /download/ api to get it from server (u get the download link in sample details) (
  edited)

<img src="https://user-images.githubusercontent.com/7339051/114413387-d9263100-9ba5-11eb-825f-ad5f9d968da8.png" width="500"/>
