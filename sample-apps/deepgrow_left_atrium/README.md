# DeepGrow for Left Atrium Segmentation on Heart Images

### Model Overview

Interactive MONAI Label App using DeepGrow (https://arxiv.org/abs/1903.08205) to label left atrium over single modality 3D MRI Images

### Running the app

monailabel start_server --app /path/on/server/to/monai_label/sample-apps/deepgrow_left_atrium/ --studies /path/to/data/

### Data

The training data is from Medical Segmentation Decathlon (http://medicaldecathlon.com/).

- Target: Left Atrium
- Task: Segmentation 
- Modality: MRI

### Input

- 3 channels (MRI + foreground points + background points)

### Output

- 1 channel representing left atrium


![DeepGrow for left atrium](../../docs/images/sample-apps/deepedit_left_atrium.png)
