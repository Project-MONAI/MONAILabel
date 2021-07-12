# DeepEdit for Spleen Segmentation

### Model Overview

Interactive MONAI Label App using DeepEdit to label spleen over CT Images

### Data

The training data is from Medical Segmentation Decathlon (http://medicaldecathlon.com/).

- Target: Spleen
- Task: Segmentation 
- Modality: CT

### Inputs

- 1 channel CT
- 3 channels (CT + foreground points + background points)

### Output

- 1 channel representing Spleen


![DeepEdit for spleen](../../docs/images/sample-apps/deepedit_spleen.png)