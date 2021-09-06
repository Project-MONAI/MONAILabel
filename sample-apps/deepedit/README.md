# DeepEdit Default App

### Model Overview

Interactive MONAI Label App using DeepEdit to label 3D Images. This includes a heuristic training planner and enhance DeepEdit transforms.

### Data

Researchers may want to use the Medical Segmentation Decathlon (http://medicaldecathlon.com/) to train and validate this algorithm.

- Target: Any organ or tumour
- Task: Segmentation 
- Modality: CT/MR

### Inputs

- 1 channel CT or MR
- 3 channels (CT or MR + foreground points + background points)

### Output

- 1 channel representing the organ of interest
