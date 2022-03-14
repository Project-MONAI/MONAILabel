import lib.infers
import lib.trainers
from monai.networks.nets import BasicUNet

PRE_TRAINED_PATH = "https://github.com/Project-MONAI/MONAILabel/releases/download/data"
NGC_PATH = "https://api.ngc.nvidia.com/v2/models/nvidia/med"

config = {
    "uri": f"{PRE_TRAINED_PATH}/pathology_deepedit_nuclei.pt",
    "dimension": 2,
    "labels": "Nuclei",
    "label_colors": {"Nuclei": (0, 255, 255)},
    "network": BasicUNet(
        spatial_dims=2,
        in_channels=5,
        out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
    ),
    "infer": {"deepedit_nuclei": lib.infers.DeepEditNuclei},
    "trainer": lib.trainers.DeepEditNuclei,
}
