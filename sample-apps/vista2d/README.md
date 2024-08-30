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

# VISTA2D Application
A reference app to run the inference task to segment cells. This app works in CellProfiler for now. All samples in the CellProfiler are provided from local path.

### Table of Contents
- [Supported Viewers](#supported-viewers)
- [Pretrained Models](#pretrained-models)
- [How To Use the App](#how-to-use-the-app)

### Supported Viewers

The VISTA2D Application supports the following viewer:

- [CellProfiler](../../plugins/cellprofiler/)

### Pretrained Models

The following are the models which are currently added into Pathology App:

| Name | Description |
|------|-------------|
| VISTA2D | An example of instance segmentation for the cell segmentation. |

<details>
    <summary><strong>Model Details (dataset, input, outputs)</strong></summary>

#### Dataset

You can use the [cellpose dataset](https://www.cellpose.org/dataset) for inference.

#### Inputs

TIFF Images

#### Output

Segmentation Masks

</details>

### How To Use the App

```bash
# skip this if you have already downloaded the app or using github repository (dev mode)
monailabel apps --download --name vista2d --output apps

# Start server with vista2d model
monailabel start_server --app apps/vista2d --studies datasets --conf models vista2d --conf preload true --conf skip_trainers true
```

**Specify bundle version** (Optional)
Above command will download the latest bundles from Model-Zoo by default. If a specific or older bundle version is used, users can add version `_v` followed by the bundle name. Example:

```bash
monailabel start_server --app apps/vista2d --studies datasets --conf models vista2d_v0.2.1 --conf preload true --conf skip_trainers true
```
