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

# Pathology Sample Application
A reference app to run inference + train tasks to segment Nuclei. This app works in both Digital Slide Archive (DSA) and QuPath Viewers, but we recommend using DSA as an endpoint for studies; however, you can also use the local datastore as a studies folder.

### Table of Contents
- [Supported Viewers](#supported-viewers)
- [Installation Requirements](#installation-requirements)
- [Pretrained Models](#pretrained-models)
- [How To Use the App](#how-to-use-the-app)
- [Performance Benchmarking](#performance-benchmarking)

### Supported Viewers
The Pathology Sample Application supports the following viewers:

- [QuPath](../../plugins/qupath/)
- [Digital Slide Archive](../../plugins/dsa/)

For more information on each of the viewers, see the [plugin extension folder](../../plugins) for the given viewer.

### Installation Requirements
MONAI Label for Pathology requires an additional dependency of OpenSlide or CuCIM. Make sure that any .dll or .so files are in the system load path.

For **Windows**, make sure `<openslide_folder>/bin` is added in PATH environment.

For **Ubuntu**: `apt install openslide-tools` or `pip install cucim`

### Pretrained Models

The following are the models which are currently added into Pathology App:

| Name | Description |
|------|-------------|
| Segmentation Nuclei | An example of multi-label segmentation for the following labels: Neoplastic, Inflammatory, Connective/Soft Tissue, Dead, and Epithelial. |
| DeepEdit Nuclei  | It is a combination of both Interaction + Auto Segmentation model, trained to segment Nuclei cells that combines all cell types from the standard segmentation model above as *Nuclei*. |
| NuClick | This is NuClick implementation (UNet model) as provided at: https://github.com/mostafajahanifar/nuclick_torch. Training task for monailabel is not yet supported. |

<details>
    <summary><strong>Model Details (dataset, input, outputs)</strong></summary>

#### Dataset

The above _Nuclei_ models are trained on [PanNuke Dataset for Nuclei Instance Segmentation and Classification](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)

#### Inputs

- WSI Images
- Region (ROI) of WSI Image

#### Output

Segmentation Mask is produced in one of the following formats:

- Standard JSON
- [DSA Document](https://digitalslidearchive.github.io/HistomicsTK/examples/segmentation_masks_to_annotations) (JSON)
- [ASAP Annotation XML](https://computationalpathologygroup.github.io/ASAP/)

</details>

### How to Use the App
The following commands are examples of how to start the Pathology Sample Application.  Make sure when you're running the command that you use the correct app and studies path for your system.

```bash
# Download Pathology App (skip this if you have already downloaded the app or using github repository (dev mode))
monailabel apps --download --name pathology --output workspace

# Start MONAI Label Server with all models
monailabel start_server --app workspace/pathology --studies datasets/wsi

# Start MONAI Label Server with all models and use specific bundle versions
monailabel start_server --app workspace/pathology --studies datasets/wsi --conf hovernet_nuclei 0.1.8 --conf nuclick 0.1.0 --conf classification_nuclei 0.1.0

# Start MONAI Label Server with HoVerNet Nuclei model
monailabel start_server --app workspace/pathology --studies datasets/wsi --conf models hovernet_nuclei

# Start MONAI Label Server with multiple models
monailabel start_server --app workspace/pathology --studies datasets/wsi

# Start MONAI Label Server with HoVerNet Nuclei model and preload on GPU
monailabel start_server --app workspace/pathology --studies datasets/wsi --conf models hovernet_nuclei --conf preload true

# Start MONAI Label Server with HoVerNet Nuclei in Inference Only mode
monailabel start_server --app workspace/pathology --studies datasets/wsi --conf models hovernet_nuclei --conf skip_trainers true
```
### Usage (Development Mode)

```bash
git clone https://github.com/Project-MONAI/MONAILabel.git
cd MONAILabel
pip install -r requirements.txt
```
Install Openslide:

> Install [Openslide](https://openslide.org/) binaries manually and make sure .dll or .so files for openslide are in system load path.
> - For windows, make sure **&lt;openslide_folder&gt;**/bin is added in PATH environment.
> - For ubuntu: `apt install openslide-tools`

#### FileSystem as Datastore

```bash
# download sample wsi image (skip this if you already have some)
mkdir sample_wsi
cd sample_wsi
wget https://demo.kitware.com/histomicstk/api/v1/item/5d5c07539114c049342b66fb/download
cd -

# run server
./monailabel/scripts/monailabel start_server --app sample-apps/pathology --studies datasets/wsi
```

### Performance Benchmarking

The performance benchmarking is done using the MONAILabel server and DSA client. All the details are captured in a [Google Sheet](https://docs.google.com/spreadsheets/d/1TeSOGzcTeeIThEvd_eflJNx0hhZiELNGBiYzwKyYEFg/edit?usp=sharing).

The following graph shows a summary of:

- NucleiDetection (CPU-Based DSA Algorithm)
- Segmentation/DeepEdit (MONAILabel models)

<table>
<tr>
<td colspan="2"><img src="../../docs/images/DSAPerf1.png"/></td>
</tr>
<tr>
<td><img src="../../docs/images/DSAPerf2.png"/></td>
<td><img src="../../docs/images/DSAPerf3.png"/></td>
</tr>
</table>

#### Digital Slide Arhive (DSA) as Datastore

###### DSA

You need to install DSA and upload some test images.
Refer: https://github.com/DigitalSlideArchive/digital_slide_archive/tree/master/devops/dsa

##### MONAILabel Server

Following are some config options:

| Name                 | Description                                                                                                                 |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| preload              | Preload models into GPU. Default is False.                                                                                  |
| roi_size             | Default ROI Size for inference in [x,y] format. Default is [1024,1024].                                                       |
| dsa_folder           | Optional. Comma seperated DSA Folder IDs. Normally it is <folder_id> of a folder under Collections where Images are stored. |
| dsa_api_key          | Optional. API Key helps to query asset store to fetch direct local path for WSI Images.                                     |
| dsa_asset_store_path | Optional. It is the DSA assetstore path that can be shared with MONAI Label server to directly read WSI Images.             |

```bash
  # run server (Example: DSA API URL is http://0.0.0.0:8080/api/v1)
  ./monailabel/scripts/monailabel start_server --app sample-apps/pathology \
    --studies http://0.0.0.0:8080/api/v1 \

  # run server (Advanced options)
  ./monailabel/scripts/monailabel start_server --app sample-apps/pathology \
    --studies http://0.0.0.0:8080/api/v1 \
    --conf dsa_folder 621e94e2b6881a7a4bef5170 \
    --conf dsa_api_key OJDE9hjuOIS6R8oEqhnVYHUpRpk18NfJABMt36dJ \
    --conf dsa_asset_store_path digital_slide_archive/devops/dsa/assetstore

```
