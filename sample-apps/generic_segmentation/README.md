# Generic Segmentation using UNet

### Model Overview

This generic App could be utilised to train a standard (non-interactive) MONAI Label App using UNet to label 3D volumes

If researchers are interested on using this App, please clone the folder and adjust the following hyperparameters:

- Network
    > This App uses the UNet as the default network. This can be changed in the **main.py** file. 
  > Researchers can define their own network or use one of the listed [here](https://docs.monai.io/en/latest/networks.html)

- Input image size
  
    > By default, this App is programmed to work on images of size (256, 256, 128). However, researchers can change this according to the GPU memory 
    their own task. This can be changed in **[./lib/infer.py](./lib/infer.py)** and **[./lib/train.py](./lib/train.py)**
  
- [Spatial and intensity transformation](https://docs.monai.io/en/latest/transforms.html) for pre and post processing
  
  > By default, this App comes with the following transforms to pre process the images:
  > - [LoadImaged](https://docs.monai.io/en/latest/_modules/monai/transforms/io/array.html#LoadImage) -> Loads both image and label from a dictionary
  > - [AddChanneld](https://docs.monai.io/en/latest/_modules/monai/transforms/utility/array.html#AddChannel) -> Add an extra channel to the image and label, 
  > - [Spacingd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Spacingd) -> Resample image and label to a different image space,
  > - [Orientationd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Orientationd) -> Reorient image and labels,
  > - [NormalizeIntensityd](https://docs.monai.io/en/latest/_modules/monai/transforms/intensity/dictionary.html#NormalizeIntensityd) ->  This transform mormalizes intensity of the image,
  > - [RandShiftIntensityd](https://docs.monai.io/en/latest/_modules/monai/transforms/intensity/dictionary.html#RandShiftIntensityd) -> This transform randomly shifts image intensity,
  > - [CropForegroundd](https://docs.monai.io/en/latest/_modules/monai/transforms/croppad/dictionary.html#CropForegroundd) -> This transform crops only the foreground object of the expected images. 
      The typical usage is to help training and evaluation if the valid part is small in the whole medical image, 
  > - [RandFlipd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#RandFlipd) -> This transfom randomly flips the image along axes,
  > - [RandAffined](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#RandAffined) -> This applies random affine transformation,
  > - [ToTensord](https://docs.monai.io/en/latest/_modules/monai/transforms/utility/dictionary.html#ToTensord) -> This converts the input image to a tensor without applying any other transformations,

- Number of epochs
  > Default value is 50 Epochs. Change this in the [info.YAML](./info.yaml) file.

- learning rate
  > Default value is 0.0001. Change this in the [info.YAML](./info.yaml) file

- validation split used during training
    > Default value is 0.2. Change this in the [info.YAML](./info.yaml) file

### Inputs

- 1 channel for the image modality

### Output

- 1 channel representing the segmented organ/tumor/tissue

### Structure of the App

- **[./lib/infer.py](./lib/infer.py)** is the script where researchers define the inference class (i.e. type of inferer, pre transforms for inference, etc).
- **[./lib/train.py](./lib/train.py)** is the script to define the pre and post transforms to train the network/model
- **[./lib/activelearning.py](./lib/activelearning.py)** is the file to define the image selection techniques.
- **[./lib/transforms.py](./lib/transforms.py)** is the file to define customised transformations to be used in the App
- **[info.yaml](./info.yaml)** is the file to define hyperparameters such as epochs, learning and validation split percentage.
- **[main.py](./main.py)** is the script to define network architecture, enable the active learning strategy, etc  



