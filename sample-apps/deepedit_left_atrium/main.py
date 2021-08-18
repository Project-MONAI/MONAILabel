import logging
import os

from lib import Deepgrow, MyStrategy, MyTrain, Segmentation
from monai.networks.nets.dynunet_v1 import DynUNetV1

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.network = DynUNetV1(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            kernel_size=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            upsample_kernel_size=[
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.download(
            [
                (
                    self.pretrained_model,
                    "https://github.com/Project-MONAI/MONAILabel/releases/download/data/deepedit_left_atrium.pt",
                ),
            ]
        )

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="DeepEdit - Left Atrium",
            description="Active learning solution using DeepEdit to label left atrium over 3D MRI Images",
            version=2,
        )

    def init_infers(self):
        return {
            "deepedit": Deepgrow([self.pretrained_model, self.final_model], self.network),
            "left_atrium": Segmentation([self.pretrained_model, self.final_model], self.network),
        }

    def init_trainers(self):
        return {"deepedit_left_atrium": MyTrain(self.model_dir, self.network, load_path=self.pretrained_model)}

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }
