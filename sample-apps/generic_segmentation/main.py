import logging
import os

from lib import MyInfer, MyStrategy, MyTrain
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.network = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.download(
            [
                (
                    self.pretrained_model,
                    "https://api.ngc.nvidia.com/v2/models/nvidia/med/"
                    "clara_pt_spleen_ct_segmentation/versions/1/files/models/model.pt",
                ),
            ]
        )

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Segmentation - Generic",
            description="Active Learning solution to label generic organ",
            version=2,
        )

    def init_infers(self):
        infers = {
            "segmentation": MyInfer([self.pretrained_model, self.final_model], self.network),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_trainers(self):
        return {"segmentation": MyTrain(self.model_dir, self.network, load_path=self.pretrained_model)}

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }
