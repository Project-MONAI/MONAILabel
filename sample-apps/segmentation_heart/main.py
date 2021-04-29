import logging
import os

from lib import MyInfer, MyTrain, MyActiveLearning
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monailabel.utils.infer.deepgrow_2d import InferDeepgrow2D
from monailabel.utils.infer.deepgrow_3d import InferDeepgrow3D
from monailabel.interface import ActiveLearning
from monailabel.interface.app import MONAILabelApp

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        model_dir = os.path.join(app_dir, "model")
        infers = {
            "deepgrow_2d": InferDeepgrow2D(os.path.join(model_dir, "deepgrow_2d.ts")),
            "deepgrow_3d": InferDeepgrow3D(os.path.join(model_dir, "deepgrow_3d.ts")),
            "segmentation_heart": MyInfer(
                path=os.path.join(model_dir, "segmentation_heart.pth"),
                network=UNet(
                    dimensions=3, in_channels=1,
                    out_channels=2, channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2, norm=Norm.BATCH, dropout=0.2)),
        }

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            active_learning=ActiveLearning()
        )

    def train(self, request):
        epochs = request.get('epochs', 1)
        amp = request.get('amp', True)
        device = request.get('device', 'cuda')
        lr = request.get('lr', 0.0001)

        logger.info(f"Training request: {request}")
        task = MyTrain(
            output_dir=os.path.join(self.app_dir, "train", "train_0"),
            data_list=self.datastore().datalist(),
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

    def next_sample(self, request):
        if request.get('strategy') == 'tta':
            myActiveLearning = MyActiveLearning(os.path.join(self.app_dir, "model", "segmentation_heart.pth") )
            return myActiveLearning(self.dataset().get_unlabeled_images())
        else:
            return super().next_sample(request)

