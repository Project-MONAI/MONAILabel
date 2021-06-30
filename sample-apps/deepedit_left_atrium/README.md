# DeepEdit for Left Atrium Segmentation on Heart Images

### Model Overview

Interactive MONAI Label App using DeepEdit to label left atrium over single modality 3D MRI Images

### Data

The training data is from Medical Segmentation Decathlon (http://medicaldecathlon.com/).

- Target: Left Atrium
- Task: Segmentation 
- Modality: MRI

### Input

Inputs: 

- 1 channel MRI 
- 3 channels (MRI + foreground points + background points)

### Output

Output: 1 channel representing left atrium


![DeepEdit for left atrium](../../docs/images/sample-apps/deepedit_left_atrium.png)