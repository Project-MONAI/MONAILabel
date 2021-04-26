import logging
import os

from lib import MyInfer, MyTrain, MyActiveLearning
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monailabel.helpers.infer.deepgrow_2d import InferDeepgrow2D
from monailabel.helpers.infer.deepgrow_3d import InferDeepgrow3D
from monailabel.interface.app import MONAILabelApp

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        model_dir = os.path.join(app_dir, "model")
        infers = {
            "deepgrow_2d": InferDeepgrow2D(os.path.join(model_dir, "deepgrow_2d.ts")),
            "deepgrow_3d": InferDeepgrow3D(os.path.join(model_dir, "deepgrow_3d.ts")),
            "segmentation_spleen": MyInfer(os.path.join(model_dir, "segmentation_spleen.ts")),
        }

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            active_learning=MyActiveLearning()
        )

    def train(self, request):
        epochs = request.get('epochs', 1)
        amp = request.get('amp', True)
        device = request.get('device', 'cuda')
        lr = request.get('lr', 0.0001)

        logger.info(f"Training request: {request}")
        task = MyTrain(
            output_dir=os.path.join(self.app_dir, "train", "train_0"),
            data_list=self.dataset().datalist(),
            network=UNet(
                dimensions=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH
            ),
            device=device,
            lr=lr
        )

        return task(max_epochs=epochs, amp=amp)
