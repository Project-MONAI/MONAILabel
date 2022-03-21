# Available MONAI Label Apps

### Overview

In this folder you will find examples of the **three available paradigms** across Radiology and Pathology use-cases: Two interactive (DeepGrow, DeepEdit) and one for automated segmentation.

## [Radiology](./radiology)

This app has example models to do both interactive and automated segmentation over radiology (3D) images.
If you are developing any examples related to radiology, you should refer this app.  It has examples for following 3 types of model.
- DeepEdit (Interactive + Auto Segmentation)
  - Spleen, Liver, Kidney, etc...
- Deepgrow (Interactive)
  - Any Organ/Tissue but pretrained to work well on Spleen, Liver, Kidney, etc...
- Segmentation (Auto Segmentation)
  - Spleen, Liver, Kidney, etc...


## [Pathology](./pathology)

This app has example models to do both interactive and automated segmentation over pathology (WSI) images.
If you are developing any examples related to pathology, you should refer this app.  It has examples for following 2 types of model.
- DeepEdit (Interactive + Auto Segmentation)
  - Nuclei Segmentation
- Segmentation (Auto Segmentation)
  - Nuclei multi-label segmentation for
    - Neoplastic cells
    - Inflammatory
    - Connective/Soft tissue cells
    - Dead Cells
    - Epithelial


### Deprecated Apps
Following apps are deprecated/removed.  It is recommended to use [Radiology](./radiology) and [Pathology](./pathology) apps for reference.

##### ~~DeepGrow~~

##### ~~DeepEdit~~

##### ~~Segmentation~~



