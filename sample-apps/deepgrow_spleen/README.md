# DeepGrow for Spleen Segmentation

### Model Overview

Interactive MONAI Label App using DeepGrow (https://arxiv.org/abs/1903.08205) to label spleen over CT Images

### Running the app

monailabel start_server --app /path/on/server/to/monai_label/sample-apps/deepgrow_spleen/ --studies /path/to/data/

### Data

The training data is from Medical Segmentation Decathlon (http://medicaldecathlon.com/).

- Target: Spleen
- Task: Segmentation 
- Modality: CT

### Input

- 3 channels (CT + foreground points + background points)

### Output

- 1 channel representing Spleen

![DeepGrow for spleen](../../docs/images/sample-apps/deepedit_spleen.png)
