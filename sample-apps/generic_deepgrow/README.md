# Generic Deepgrow

### Model Overview

This MONAI Label App is based on DeepGrow which allows for an interactive segmentation, where 
the user can guide the segmentation using positive and negative clicks (https://arxiv.org/abs/1903.08205).
It uses pre-trained Deepgrow Models for NVIDIA Clara.

Example data inference can be tested on CT based datasets from [medical segmentation decathlon](http://medicaldecathlon.com/). The data from Liver, Spleen and Pancreas tasks are all CT-based datasets 

To those extending this App, use the below command through CLI and adjust the following hyper-parameters in the codebase:

```bash
monailabel apps --name generic_deepgrow --download --output myapp
```

- Network
  
  > The network definition can be change in the **main.py** file. Currently, it uses U-net based network with 
  > residual blocks and has 32 channels with 5 levels of encoding. We do not recommend changing the structure as pre-trained 
  > weights are available for usage with the defined network structure.
  > Researchers can define their own network or use one of the listed [here](https://docs.monai.io/en/latest/networks.html) 
  
- Input image size
  
    > By default, this App is programmed to work on images of size (128, 192, 192) for DeepGrow 3D and (256, 256) for DeepGrow 2D. However, researchers can change this according to the GPU memory for
    their own task. This can be changed in **[./main.py](./main.py)**. We do not recommend changing the size for DeepGrow 2D however the size for DeepGrow 3D should be adjusted depending upon the object of interest.
 
- [Spatial and intensity transformation](https://docs.monai.io/en/latest/transforms.html) for pre and post processing
  
  > By default, this App comes with the following transforms to pre process the images and simulate clicks:
  > - [LoadImaged](https://docs.monai.io/en/latest/_modules/monai/transforms/io/array.html#LoadImage) -> Loads both image and label from a dictionary
  > - [AsChannelFirstd](https://docs.monai.io/en/latest/transforms.html#aschannelfirstd) -> Transforms both image and label to have the channel as first dimension
  > - [Spacingd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Spacingd) -> Resample image and label to a different image space,
  > - [Orientationd](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Orientationd) -> Reorient image and labels,
  > - [AddChanneld](https://docs.monai.io/en/latest/_modules/monai/transforms/utility/array.html#AddChannel) -> Add an extra channel to the image and label,
  > - [SpatialCropForegroundd](https://docs.monai.io/en/latest/transforms.html#cropforegroundd) -> This transform randomly changes image intensity by a gamma value, 
  > - [Resized](https://docs.monai.io/en/latest/_modules/monai/transforms/spatial/dictionary.html#Resized) -> Resample image to a different size,
  > - [NormalizeIntensityd](https://docs.monai.io/en/latest/_modules/monai/transforms/intensity/dictionary.html#NormalizeIntensityd) -> Normalize intensity of the image,


  Transformations used for the clicks simulation [(DeepGrow App)](https://docs.monai.io/en/latest/apps.html)
  > 
  > - [FindAllValidSlicesd](https://docs.monai.io/en/latest/_modules/monai/apps/deepgrow/transforms.html#FindAllValidSlicesd) -> This transform finds/lists all valid slices in the label. Label is assumed to be a 4D Volume with shape CDHW, where C=1,
  > - [AddInitialSeedPointd](https://docs.monai.io/en/latest/_modules/monai/apps/deepgrow/transforms.html#AddInitialSeedPointd) -> This transform adds random guidance as initial seed point for a given label,
  > - [AddGuidanceSignald](https://docs.monai.io/en/latest/_modules/monai/apps/deepgrow/transforms.html#AddGuidanceSignald) -> This transform adds guidance signal (foreground and background clicks) for input image, 
  
- Number of epochs
  > Default value is 50 Epochs for DeepGrow 3D and 20 Epochs for DeepGrow 2D. Change this in the [info.YAML](./info.yaml) file.

- learning rate
  > Default value is 0.0001. Change this in the [info.YAML](./info.yaml) file

- validation split used during training
    > Default value is 0.2. Change this in the [info.YAML](./info.yaml) file

### Inputs

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
 
### Data

- Target: Any organ
- Task: Segmentation
- Modality: MRI or CT

### Input

- 3 channels (CT + foreground points + background points)

### Output

- 1 channel representing the segmented organ/tissue/tumor

## Links

- https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_deepgrow_3d_annotation
- https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_deepgrow_2d_annotation
