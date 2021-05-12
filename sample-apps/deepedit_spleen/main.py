import logging
import os

from lib import Deepgrow, MyStrategy, MyTrain, Segmentation
from monai.networks.nets import DynUNet

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import TTA, Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")

        infers = {
            "deepgrow": Deepgrow(os.path.join(self.model_dir, "deep_edit.ts")),
            "segmentation": Segmentation(os.path.join(self.model_dir, "deep_edit.ts")),
        }

        strategies = {
            "random": Random(),
            "first": MyStrategy(),
            "tta": TTA(os.path.join(self.model_dir, "deep_edit.ts")),
        }

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            strategies=strategies,
        )

    def train(self, request):
        name = request.get("name", "model_01")
        epochs = request.get("epochs", 1)
        amp = request.get("amp", True)
        device = request.get("device", "cuda")
        lr = request.get("lr", 0.0001)
        val_split = request.get("val_split", 0.2)

        logger.info(f"Training request: {request}")
        task = MyTrain(
            output_dir=os.path.join(self.model_dir, name),
            data_list=self.datastore().datalist(),
            network=DynUNet(
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
            ),
            device=device,
            lr=lr,
            val_split=val_split,
        )

        return task(max_epochs=epochs, amp=amp)
