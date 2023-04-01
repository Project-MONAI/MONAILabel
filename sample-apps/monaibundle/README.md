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

# MONAI Bundle Application
The MONAIBundle App allows you to easily pull any MONAI Bundle from the [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models) and import it into MONAI Label. However, it's important to note that any MONAI Bundle used with MONAI Label must meet the following constraints:

- It must comply with the [MONAI Bundle Specification](https://docs.monai.io/en/latest/mb_specification.html).
- For inference, the bundle must define either an `inference.json` or `inference.yaml` file, and it must include the keys described in the bundle.py file located in the `monailabel/tasks/infer/` directory.
- For training, the bundle must define either a `train.json` or `train.yaml file`, and it must include the keys described in the bundle.py file located in the `monailabel/tasks/train/` directory.
- For multi-GPU training, the bundle must define either a `multi_gpu_train.json` or `multi_gpu_train.yaml` file.

These constraints ensure that any MONAI Bundle used with MONAI Label is compatible with the platform and can be seamlessly integrated into your workflow.

### Table of Contents
- [Supported Models](#supported-models)
- [How To Use the App](#how-to-use-the-app)
- [Epistemic Scoring using MONAI Bundles](#epistemic-Scoring-for-monaibundle-app)

### Supported Models


The MONAIBundle App currently supports most labeling models in the Model-Zoo. You can find a table of supported labeling tasks below. Please note that the list of supported tasks is updated based on the latest release from the [Model-Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models).

| Bundle | Model | Objects | Modality | Note |
|:----:|:-----:|:-------:|:--------:|:----:|
| [spleen_ct_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation) | UNet | Spleen | CT | A model for (3D) segmentation of the spleen |
| [swin_unetr_btcv_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/swin_unetr_btcv_segmentation) | SwinUNETR | Multi-Organ | CT | A model for (3D) multi-organ segmentation |
| [prostate_mri_anatomy](https://github.com/Project-MONAI/model-zoo/tree/dev/models/prostate_mri_anatomy) | UNet | Prostate | MRI | A model for (3D) prostate segmentation from MRI image |
| [pancreas_ct_dints_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/pancreas_ct_dints_segmentation) | DiNTS | Pancreas/Tumor | CT | An automl method for (3D) pancreas/tumor segmentation |
| [renalStructures_UNEST_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/renalStructures_UNEST_segmentation) | UNesT | Kidney Substructure | CT |  A pre-trained for inference (3D) kidney cortex/medulla/pelvis segmentation |
| [wholeBrainSeg_UNEST_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBrainSeg_Large_UNEST_segmentation) | UNesT | Whole Brain | MRI T1 |  A pre-trained for inference (3D) 133 whole brain structures segmentation |
| [spleen_deepedit_annotation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_deepedit_annotation) | DeepEdit | Spleen| CT | An interactive method for 3D spleen Segmentation |


**Note:** The MONAIBundle app uses the MONAI Bundle API to retrieve information about the latest models from the Model-Zoo. If you're encountering rate limiting issues while using the app, you can input your personal access token using the --conf auth_token command. For more information on rate limiting and how to generate an access token, please refer to the following link: https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting

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

# Skip Training Tasks or Infer only mode
monailabel start_server --app workspace/monaibundle --studies workspace/images --conf models spleen_ct_segmentation_v0.1.0 --conf skip_trainers true
```

### Epistemic Scoring for monaibundle app
The MONAIBundle App supports epistemic scoring using bundle models. To use epistemic scoring, you can specify a valid scoring bundle model as the `epistemic_model` configuration parameter when running the app, E.g., `--conf epistemic_model <bundlename>`. If a valid scoring bundle model is provided, scoring inference will be triggered. The loaded scoring bundle model can be either a model from the Model Zoo or a local bundle, but it must support the `dropout` argument.

With epistemic scoring, MONAIBundle can provide measures of uncertainty or confidence in the model's predictions, which can be useful in a variety of applications.

```bash
# Use the UNet in spleen_ct_segmentation_v0.2.0 bundle as epistemic scoring model.
# Manual define epistemic scoring parameters
monailabel start_server \
  --app workspace/monaibundle \
  --studies workspace/images \
  --conf models spleen_ct_segmentation_v0.2.0,swin_unetr_btcv_segmentation_v0.2.0 \
  --conf epistemic_model spleen_ct_segmentation_v0.2.0
  --conf epistemic_max_samples 0 \
  --conf epistemic_simulation_size 5
  --conf epistemic_dropout 0.2

```

#### Additional Configs

To set configuration parameters for MONAI Label Server, use the `--conf <name> <value>` flag followed by the parameter name and value while starting the MONAI Label Server.

| Name                      | Values          | Description                                                                                 |
|---------------------------|-----------------|---------------------------------------------------------------------------------------------|
| zoo_source                | string          | _Default value:_ github                                                                     |
| zoo_repo                  | string          | _Default value:_ Project-MONAI/model-zoo/hosting_storage_v1                                 |
| preload                   | true, **false** | Preload model into GPU                                                                      |
| skip_trainers             | true, **false** | Skip adding training tasks (Run in Infer mode only)                                         |
| epistemic_max_samples     | int             | _Default value:_ 0    ;  Epistemic scoring parameters                                       |
| epistemic_simulation_size | int             | _Default value:_ 5    ;  Epistemic simulation size parameters                               |
| epistemic_dropout         | float           | _Default value:_ 0.2  ;  Epistemic scoring parameters: Dropout rate for scoring models      |                               |
