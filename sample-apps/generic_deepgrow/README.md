# Generic Deepgrow

### Model Overview

Interactive MONAI Label App using DeepGrow (https://arxiv.org/abs/1903.08205).
It uses pre-trained Deepgrow Models for NVIDIA Clara.

### Data

- Target: Any organ
- Task: Segmentation
- Modality: MRI or CT

### Input

- 3 channels (CT + foreground points + background points)

### Output

- 1 channel representing the segmented organ/tissue/tumor

## Links

- https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_deepgrow_3d_annotation
- https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_deepgrow_2d_annotation
