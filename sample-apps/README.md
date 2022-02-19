# Available MONAI Label Apps

### Overview

In this folder you will find examples of the **three available paradigms**: Two interactive (DeepGrow, DeepEdit) and one for automated segmentation.


#### DeepGrow

Users that want to build deepgrow-based Apps, please refer to the [deepgrow App](./deepgrow). 

This app is based on the work presented by [Sakinis, Tomas, et al. "Interactive segmentation of medical images through fully convolutional neural networks." arXiv preprint arXiv:1903.08205 (2019)](https://arxiv.org/abs/1903.08205).



#### DeepEdit

Similar to the deepgrow App, you'll find a generic [deepedit](./deepedit) that researchers can use to build their own deepedit-based app for single label segmentaion tasks.

Users that want to work on multiple label segmentation tasks, please refer to the [deepedit_multilabel](./deepedit_multilabel) 


#### Automated Segmentation

As the deepgrow and deepedit Apps, researchers can try the non-interactive Apps for [spleen](./segmentation_spleen) and [left atrium](./segmentation_left_atrium) using UNet. There is also the generic segmentation App that researchers can clone to create their own App. 


More examples of these Apps can be found in the [MONAI Label Apps Zoo](https://github.com/diazandr3s/MONAILabel-Apps)




