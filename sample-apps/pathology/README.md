# Pathalogy Use Case
> _This App is currently under active development._

### Overview
This is a reference app to run infer + train tasks to segment Nuclei. It comes with follwoing 2 pre-trained weights/model (UNET). 
 - Segmentation - This show-cases an example for multi-label segmentation.  It tries segment following labels.
   - Neoplastic cells
   - Inflammatory 
   - Connective/Soft tissue cells
   - Dead Cells
   - Epithelial
 - DeepEdit - It is a combination of both [Interaction + Auto Segmentation](https://github.com/Project-MONAI/MONAILabel/wiki/DeepEdit) model which is trained to segment Nuclei cells that combines all above labels as *Nuclei*.

### Dataset
Above models are trained on [PanNuke Dataset for Nuclei Instance Segmentation and Classification](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)

### Inputs
- WSI Images
- Region (ROI) of WSI Image

### Output
Segmentation Mask are produced in one of the following formats
 - Standard JSON
 - [DSA Document](https://digitalslidearchive.github.io/HistomicsTK/examples/segmentation_masks_to_annotations) (JSON)
 - [ASAP Annotation XML](https://computationalpathologygroup.github.io/ASAP/)
 
### Usage (Development Mode)
We recommend to use Digital Slide Archive as endpoint for studies.  However you can also also use FileSystem as studies folder.

```bash
git clone https://github.com/Project-MONAI/MONAILabel.git
cd MONAILabel
pip install -r requirements.txt
```

#### FileSystem as Datastore

```bash
  # download sample wsi image (skip this if you already have some)
  mkdir sample_wsi
  cd sample_wsi
  wget https://demo.kitware.com/histomicstk/api/v1/item/5d5c07539114c049342b66fb/download
  cd -
  
  # run server
  ./monailabel/scripts/monailabel start_server --app apps/pathology --studies datasets/wsi
  
  # run wsi inference api
  # visit http://127.0.0.1:8000/#/Infer/api_run_wsi_inference_infer_wsi__model__post
  
  # Prototype QuPath is available at: https://github.com/SachidanandAlle/qupath
  # Build and Run QuPath
  # 1. Open sample wsi image from sample_wsi (shared with MONAILabel)
  # 2. Ctrl + CLick on some area of Image to add MONAI specific ROI
  # 3. Resize ROI if needed
  # 4. Menu -> MONAILabel -> Run
  
```
 
#### Digital Slide Arhive (DSA) as Datastore
##### DSA
  You need to install DSA and upload some test images.  Refer: https://github.com/DigitalSlideArchive/digital_slide_archive/tree/master/devops/dsa

##### MONAILabel Server
Following are some config options:

| Name                 | Description                                                                                                      |
|----------------------|------------------------------------------------------------------------------------------------------------------|
| dsa_folder           | DSA Folder ID. Normally it is <folder_id> of a folder under Collections where Images are stored.                 |
| dsa_api_key          | Optional. API Key helps to query asset store to fetch direct local path for WSI Images.                          |
| dsa_asset_store_path | Optional.  It is the DSA assetstore path that can be shared with MONAI Label server to directly read WSI Images. |

```bash  
  # run server (Example: DSA API URL is http://0.0.0.0:8080/api/v1)
  ./monailabel/scripts/monailabel start_server --app apps/pathology \
    --studies http://0.0.0.0:8080/api/v1 \
    --conf dsa_folder 621e94e2b6881a7a4bef5170 \
    --conf dsa_api_key OJDE9hjuOIS6R8oEqhnVYHUpRpk18NfJABMt36dJ \
    --conf dsa_asset_store_path digital_slide_archive/devops/dsa/assetstore
  
```

##### DSA Client Plugin
```bash
  cd plugins/dsa
  docker build -t projectmonai/monailabel/dsa:latest .
```

Upload new Task (under Slicer CLI Web Tasks) using the above docker image.  This will add/enable MONAILabel under Analysis Page.