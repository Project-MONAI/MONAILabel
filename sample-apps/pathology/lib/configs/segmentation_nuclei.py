import lib.infers
import lib.trainers
from monai.networks.nets import BasicUNet

PRE_TRAINED_PATH = "https://github.com/Project-MONAI/MONAILabel/releases/download/data"
NGC_PATH = "https://api.ngc.nvidia.com/v2/models/nvidia/med"

config = {
    "uri": f"{PRE_TRAINED_PATH}/pathology_segmentation_nuclei.pt",
    "dimension": 2,
    "labels": {
        "Neoplastic cells": 1,
        "Inflammatory": 2,
        "Connective/Soft tissue cells": 3,
        "Dead Cells": 4,
        "Epithelial": 5,
    },
    "label_colors": {
        "Neoplastic cells": (255, 0, 0),
        "Inflammatory": (255, 255, 0),
        "Connective/Soft tissue cells": (0, 255, 0),
        "Dead Cells": (0, 0, 0),
        "Epithelial": (0, 0, 255),
    },
    "network": BasicUNet(spatial_dims=2, in_channels=3, out_channels=6),
    "infer": lib.infers.SegmentationNuclei,
    "trainer": lib.trainers.SegmentationNuclei,
}
