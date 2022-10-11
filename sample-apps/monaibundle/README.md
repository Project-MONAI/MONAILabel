<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Integration with MONAI Bundle
The App helps to pull any MONAI Bundle from [MONAI ZOO](https://github.com/Project-MONAI/model-zoo/tree/dev/models).
However the following constraints has to be met for any monai bundle to directly import and use in MONAI Label.
 - Has to meet [MONAI Bundle Specification](https://docs.monai.io/en/latest/mb_specification.html).
 - For Inference, the bundle has defined **inference.json** or **inference.yaml** and defines [these keys](../../monailabel/tasks/infer/bundle.py)
 - For Training, the bundle has defined **train.json** or **train.yaml** and defines [these keys](../../monailabel/tasks/train/bundle.py)
 - For Multi-GPU Training, the bundle has defined **multi_gpu_train.json** or **multi_gpu_train.yaml**

> By default models are picked from https://github.com/Project-MONAI/model-zoo/blob/dev/models/model_info.json

### Structure of the App

- **[lib/infers](./lib/infers)** is to define and activate inference task over monai-bundle.
- **[lib/trainers](./lib/trainers)** is to define and activate training task over monai-bundle for single/multi gpu.
- **[lib/activelearning](./lib/activelearning)** is the module to define the image selection techniques.
- **[main.py](./main.py)** is the script to extend [MONAILabelApp](../../monailabel/interfaces/app.py) class

> Modify Constants defined in [Infer](../../monailabel/tasks/infer/bundle.py) and [Train](../../monailabel/tasks/train/bundle.py) to customize and adopt if the basic standard/schema is not met for your bundle.

### Overview


### Supported Models

The Bundle App supports most labeling models in the Model Zoo, please see the table for labeling tasks.


| Bundle | Model | Objects | Modality | Note |
|:----:|:-----:|:-------:|:--------:|:----:|
| [spleen_ct_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation) | UNet | Spleen | CT | A model for (3D) segmentation of the spleen |
| [swin_unetr_btcv_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/swin_unetr_btcv_segmentation) | SwinUNETR | Multi-Organ | CT | A model for (3D) multi-organ segmentation |
| [prostate_mri_anatomy](https://github.com/Project-MONAI/model-zoo/tree/dev/models/prostate_mri_anatomy) | UNet | Prostate | MRI | A model for (3D) prostate segmentation from MRI image |
| [pancreas_ct_dints_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/pancreas_ct_dints_segmentation) | DiNTS | Pancreas/Tumor | CT | An automl method for (3D) pancreas/tumor segmentation |
| [renalStructures_UNEST_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/renalStructures_UNEST_segmentation) | UNesT | Kidney Substructure | CT |  A pre-trained for inference (3D) kidney cortex/medulla/pelvis segmentation |
| [wholeBrainSeg_UNEST_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBrainSeg_Large_UNEST_segmentation) | UNesT | Whole Brain | MRI T1 |  A pre-trained for inference (3D) 133 whole brain structures segmentation |
| [brats_mri_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation) | SegResNet | Brain Tumor | MRI |  A pre-trained for brain tumor subregions segmentation |
| [spleen_deepedit_annotation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_deepedit_annotation) | DeepEdit | Spleen| CT | An interactive method for 3D spleen Segmentation |


Supported tasks update based on [Model-Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models) release.

### How To Use?
```bash
# skip this if you have already downloaded the app or using github repository (dev mode)
monailabel apps --download --name monaibundle --output workspace

# List all available models from zoo
monailabel start_server --app workspace/monaibundle --studies workspace/images

# Pick spleen_ct_segmentation_v0.1.0 model
monailabel start_server --app workspace/monaibundle --studies workspace/images --conf models spleen_ct_segmentation_v0.1.0

# Pick spleen_ct_segmentation_v0.1.0 model and preload
monailabel start_server --app workspace/monaibundle --studies workspace/images --conf models spleen_ct_segmentation_v0.1.0 --conf preload true

# Pick DeepEdit And Segmentation model (multiple models)
monailabel start_server --app workspace/monaibundle --studies workspace/images --conf models "spleen_ct_segmentation_v0.1.0,spleen_deepedit_annotation_v0.1.0"

# Pick All (Skip Training Tasks or Infer only mode)
monailabel start_server --app workspace/monaibundle --studies workspace/images --conf models all --conf skip_trainers true
```



#### Additional Configs
Pass them as **--conf _name_ _value_** while starting MONAILabelServer

| Name          | Values          | Description                                                                                 |
|---------------|-----------------|---------------------------------------------------------------------------------------------|
| zoo_info      | string          | _Default value:_ https://github.com/Project-MONAI/model-zoo/blob/dev/models/model_info.json |
| zoo_source    | string          | _Default value:_ github                                                                     |
| zoo_repo      | string          | _Default value:_ Project-MONAI/model-zoo/hosting_storage_v1                                 |
| preload       | true, **false** | Preload model into GPU                                                                      |
| skip_trainers | true, **false** | Skip adding training tasks (Run in Infer mode only)                                         |
