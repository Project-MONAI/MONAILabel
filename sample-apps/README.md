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
Following apps shall be deprecated.  It is recommended to use [Radiology](./radiology) and [Pathology](./pathology) apps for reference.

##### DeepGrow

Users that want to build deepgrow-based Apps, please refer to the [deepgrow App](./deepgrow). 
This app is based on the work presented by [Sakinis, Tomas, et al. "Interactive segmentation of medical images through fully convolutional neural networks." arXiv preprint arXiv:1903.08205 (2019)](https://arxiv.org/abs/1903.08205).



##### DeepEdit

Similar to the deepgrow App, you'll find a generic [deepedit](./deepedit) that researchers can use to build their own deepedit-based app for single label segmentaion tasks.
Users that want to work on multiple label segmentation tasks, please refer to the [deepedit_multilabel](./deepedit_multilabel) 


##### Segmentation

As the deepgrow and deepedit Apps, researchers can try the non-interactive Apps for [spleen](./segmentation_spleen) and [left atrium](./segmentation_left_atrium) using UNet. There is also the generic segmentation App that researchers can clone to create their own App. 

More examples of these Apps can be found in the [MONAI Label Apps Zoo](https://github.com/Project-MONAI/MONAILabel/tree/apps/sample-apps)




