import logging
import os

from lib import MyTrain
from monai.networks.layers import Norm
from monai.networks.nets import BasicUNet, UNet
from monailabel.engines.infer import Deepgrow2D, Deepgrow3D, DeepgrowPipeline, SegmentationHeart
from monailabel.interface.app import MONAILabelApp

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        model_dir = os.path.join(app_dir, "model")
        features = (32, 64, 128, 256, 512, 32)
        heart = SegmentationHeart(
            path=os.path.join(model_dir, "segmentation_heart.pt"),
            network=BasicUNet(dimensions=3, in_channels=3, out_channels=1, features=features)
            # network=UNet(
            #     dimensions=3, in_channels=3,
            #     out_channels=2, channels=(16, 32, 64, 128, 256),
            #     strides=(2, 2, 2, 2),
            #     num_res_units=2, norm=Norm.BATCH, dropout=0.2)
        )
        deepgrow_3d = Deepgrow3D(os.path.join(model_dir, "deepgrow_3d.ts"))

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers={
                "deepgrow_2d": Deepgrow2D(os.path.join(model_dir, "deepgrow_2d.ts")),
                "deepgrow_3d": deepgrow_3d,
                "segmentation_heart": heart,
                "deepgrow_pipeline": DeepgrowPipeline(os.path.join(model_dir, "deepgrow_2d.ts"), deepgrow_3d),
                "deepgrow_pipeline_heart": DeepgrowPipeline(os.path.join(model_dir, "deepgrow_2d.ts"), heart),
            }
        )

    def info(self):
        return super().info()

    def infer(self, request):
        return super().infer(request)

    def train(self, request):
        epochs = request['epochs']
        amp = request.get('amp', False)
        device = request.get('device', 'cuda')
        lr = request.get('lr', 0.0001)

        logger.info(f"Training request: {request}")
        engine = MyTrain(
            output_dir=os.path.join(self.app_dir, "model", "train_0"),
            data_list=os.path.join(self.studies, "dataset.json"),
            data_root=self.studies,
            device=device,
            lr=lr
        )

        stats = engine.run(max_epochs=epochs, amp=amp)
        return stats

    def next_sample(self, request):
        return super().next_sample(request)

    def save_label(self, request):
        return super().save_label(request)
