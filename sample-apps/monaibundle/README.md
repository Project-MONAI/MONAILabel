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
 - For Inference, the bundle has defined **inference.json** or **inference.yaml** and defines [these keys](./lib/infers/bundle.py)
 - For Training, the bundle has defined **train.json** or **train.yaml** and defines [these keys](./lib/trainers/bundle.py)
 - For Multi-GPU Training, the bundle has defined **multi_gpu_train.json** or **multi_gpu_train.yaml**

> By default models are picked from https://github.com/Project-MONAI/model-zoo/blob/dev/models/model_info.json

### Structure of the App

- **[lib/infers](./lib/infers)** is to define and activate inference task over monai-bundle.
- **[lib/trainers](./lib/trainers)** is to define and activate training task over monai-bundle for single/multi gpu.
- **[lib/activelearning](./lib/activelearning)** is the module to define the image selection techniques.
- **[main.py](./main.py)** is the script to extend [MONAILabelApp](../../monailabel/interfaces/app.py) class

> Modify Constants defined in [Infer](./lib/infers/bundle.py) and [Train](./lib/trainers/bundle.py) to customize and adopt if the basic standard/schema is not met for your bundle.

### Overview


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

# Pick All
monailabel start_server --app workspace/monaibundle --studies workspace/images --conf models all

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
