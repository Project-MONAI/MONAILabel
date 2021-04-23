import logging
import os

from monai.networks.layers import Norm
from monai.networks.nets import UNet, BasicUNet
from monailabel.engines.infer import (
    InferDeepgrow2D,
    InferDeepgrow3D,
    InferDeepgrowPipeline,
    InferSegmentationHeart
)
from monailabel.engines.train import TrainSegmentationHeart
from monailabel.interface import ActiveLearning
from monailabel.interface.app import MONAILabelApp

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        model_dir = os.path.join(app_dir, "model")
        # TODO:: Remove Network and use torch.jit model
        heart = InferSegmentationHeart(
            path=os.path.join(model_dir, "segmentation_heart.pth"),
            network=UNet(
                dimensions=3, in_channels=1,
                out_channels=2, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2, norm=Norm.BATCH, dropout=0.2))
        deepgrow_3d = InferDeepgrow3D(os.path.join(model_dir, "deepgrow_3d.ts"))

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers={
                "deepgrow": InferDeepgrowPipeline(os.path.join(model_dir, "deepgrow_2d.ts"), deepgrow_3d),
                "deepgrow_2d": InferDeepgrow2D(os.path.join(model_dir, "deepgrow_2d.ts")),
                "deepgrow_3d": deepgrow_3d,
                "segmentation_heart": heart,
            },
            active_learning=ActiveLearning()
        )

    def train(self, request):
        epochs = request.get('epochs', 1)
        amp = request.get('amp', True)
        device = request.get('device', 'cuda')
        lr = request.get('lr', 0.0001)
        if request.get('network', "UNet") == "BasicUNet":
            network = BasicUNet(dimensions=3, in_channels=1, out_channels=2, features=(16, 32, 64, 128, 256, 16))
        else:
            network = UNet(
                dimensions=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH
            )

        logger.info(f"Training request: {request}")
        engine = TrainSegmentationHeart(
            output_dir=os.path.join(self.app_dir, "train", "train_0"),
            data_list=self.dataset().datalist(),
            network=network,
            device=device,
            lr=lr
        )

        return engine.run(max_epochs=epochs, amp=amp)
