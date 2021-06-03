# MONAI Label with User Scribbles based post processing
This is an initial prototype for incorporating user scribbles into a post processing method that can be used to improve segmentations from a deep learning model.

The flow of this application is outlined below:
![monailabel_crf](./docs/monailabel_crf.png)

In the diagram, the neural network (inference) stage is run only once for a given sample. The logits for this run are saved and used throughout scribble based updates to the same sample.

# Installing pre-requisites

## 1. Install MONAI with BUILD_MONAI=1
Uses CRF layer from MONAI, which requires compiling the C++/CUDA code following instructions from [MONAI docs](https://docs.monai.io/en/latest/installation.html#option-1-as-a-part-of-your-system-wide-module). 
This can be done by uninstalling any previous monai/monai-weekly version and running the following command:

`BUILD_MONAI=1 pip install git+https://github.com/Project-MONAI/MONAI#egg=monai`

Another way is use docker for MONAI

`docker run --gpus all --rm -ti --ipc=host --net=host -v /xyz:/workspace projectmonai/monai:latest`


## 2. Install SimpleCRF
Current application provides an option to switch to SimpleCRF library for doing the CRF part. This can be installed as:

`pip install simplecrf`

# Running the app

## Short App Demo
![scribble_ui](./docs/scribble_ui.gif)

## Server
On the server side, run server app using the following command:

`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../. python main.py run --app ../sample-apps/segmentation_spleen_postproc/ --studies /path/to/dataset/Task09_Spleen/imagesTrSmall/`

## Client
On the client side, run slicer and load monailabel extension:

1. Click *Next Sample" to load a sample with its initial segmentation
2. Scribbles functionality is inside *Post Processing Scribbles* section
3. To add scribbles select *Painter* or *Eraser* Tool and appropriate layer *Foreground* or *Background*
4. Painting/Erasing tool will be activated, add scribbles to each slice/view
5. Once done, click *Update* to send scribbles to server for applying the selected post processing method

Further help on setting up monailabel apps can be found in the [main README document](../../README.md).