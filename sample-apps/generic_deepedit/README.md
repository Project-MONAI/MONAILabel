# Generic DeepEdit

### Model Overview

This MONAI Label App is based on DeepEdit: an algorithm that combines
the capabilities of multiple models into one, allowing for both interactive and automated segmentation.

If researchers are interested on using this App, please clone the folder and adjust the following hyperparameters:

- Network
    > This App uses the DynUNetV1 as the default network. This can be changed in the **main.py** file. 
  > Researchers can define their own network or use one of the listed [here](https://docs.monai.io/en/latest/networks.html) 

- Input image size
  
    > By default, this App is programmed to work on images of size (128, 128, 128). However, researchers can change this according to the GPU memory 
    their own task. This can be changed in **[./lib/infer.py](./lib/infer.py)** and **[./lib/train.py](./lib/train.py)**
  
- [Spatial and intensity transformation](https://docs.monai.io/en/latest/transforms.html) for pre and post processing
  
  > By default, this App comes with the following transforms to pre process the images and simulate clicks:
  > - [LoadImaged](https://docs.monai.io/en/latest/_modules/monai/transforms/io/array.html#LoadImage) -> Loads both image and label from a dictionary
  > - [RandZoomd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#RandZoomd) -> Random zoom to the image and label,
  > - [AddChanneld](https://docs.monai.io/en/latest/_modules/monai/transforms/utility/array.html#AddChannel) -> Add an extra channel to the image and label, 
  > - [Spacingd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Spacingd) -> Resample image and label to a different image space,
  > - [Orientationd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Orientationd) -> Reorient image and labels,
  > - [NormalizeIntensityd](https://docs.monai.io/en/latest/_modules/monai/transforms/intensity/dictionary.html#NormalizeIntensityd) -> Normalize intensity of the image,
  > - [RandAdjustContrastd](https://docs.monai.io/en/latest/_modules/monai/transforms/intensity/dictionary.html#RandAdjustContrastd) -> This transform randomly changes image intensity by a gamma value, 
  > - [RandHistogramShiftd](https://docs.monai.io/en/latest/_modules/monai/transforms/intensity/dictionary.html#RandHistogramShiftd) -> This transfoms applies random nonlinear transform the the image's intensity histogram,
  > - [Resized](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Resized) -> Resample image to a different size,


  Transformations used for the clicks simulation [(DeepGrow App)](https://docs.monai.io/en/latest/apps.html)
  > 
  > - [FindAllValidSlicesd](https://docs.monai.io/en/latest/_modules/monai/apps/deepgrow/transforms.html#FindAllValidSlicesd) -> This transform finds/lists all valid slices in the label. Label is assumed to be a 4D Volume with shape CDHW, where C=1,
  > - [AddInitialSeedPointd](https://docs.monai.io/en/latest/_modules/monai/apps/deepgrow/transforms.html#AddInitialSeedPointd) -> This transform adds random guidance as initial seed point for a given label,
  > - [AddGuidanceSignald](https://docs.monai.io/en/latest/_modules/monai/apps/deepgrow/transforms.html#AddGuidanceSignald) -> This transform adds guidance signal (foreground and background clicks) for input image, 
  > - DiscardAddGuidanced -> This transform discards foreground and background tensors with a probability define as an argument. 
      > This transform is responsible for the training of both automated model and interactive model. 
      > This means, sometimes the input is a tensor with only the image and two empty tensors and sometimes is a tensor with the image and the foreground and backgounr points. 

- Number of epochs
  > Default value is 50 Epochs. Change this in the [info.YAML](./info.yaml) file.

- learning rate
  > Default value is 0.0001. Change this in the [info.YAML](./info.yaml) file

- validation split used during training
    > Default value is 0.2. Change this in the [info.YAML](./info.yaml) file

### Inputs

- 1 channel for the image modality -> Automated mode
- 3 channels (image modality + foreground points + background points) -> Interactive mode

### Output

- 1 channel representing the segmented organ/tumor/tissue

### Structure of the App

- **[./lib/infer.py](./lib/infer.py)** is the script where researchers define the inference class (i.e. type of inferer, pre transforms for inference, etc).
- **[./lib/train.py](./lib/train.py)** is the script to define the pre and post transforms to train the network/model
- **[./lib/activelearning.py](./lib/activelearning.py)** is the file to define the image selection techniques.
- **[./lib/transforms.py](./lib/transforms.py)** is the file to define customised transformations to be used in the App
- **[info.yaml](./info.yaml)** is the file to define hyperparameters such as epochs, learning and validation split percentage.
- **[main.py](./main.py)** is the script to define network architecture, enable the active learning strategy, etc



