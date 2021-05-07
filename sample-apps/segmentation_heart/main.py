import logging
import os

from lib import MyInfer, MyTrain, MyActiveLearning
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monailabel.interfaces.app import MONAILabelApp
from monailabel.utils.infer.deepgrow_2d import InferDeepgrow2D
from monailabel.utils.infer.deepgrow_3d import InferDeepgrow3D

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")

        infers = {
            "deepgrow_2d": InferDeepgrow2D(os.path.join(self.model_dir, "deepgrow_2d.ts")),
            "deepgrow_3d": InferDeepgrow3D(os.path.join(self.model_dir, "deepgrow_3d.ts")),
            "segmentation_heart": MyInfer(os.path.join(self.model_dir, "segmentation_heart.ts"))
        }

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            active_learning=MyActiveLearning(os.path.join(self.model_dir, "segmentation_heart.ts"))
        )

    def train(self, request):
        name = request.get('name', 'model_01')
        epochs = request.get('epochs', 1)
        amp = request.get('amp', True)
        device = request.get('device', 'cuda')
        lr = request.get('lr', 0.0001)
        val_split = request.get('val_split', 0.2)

        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, name)

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        load_path = load_path if os.path.exists(load_path) else os.path.join(self.model_dir, "segmentation_heart.pt")

        task = MyTrain(
            output_dir=output_dir,
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
            load_path=load_path,
            device=device,
            lr=lr,
            val_split=val_split
        )

        # TODO:: Publish model to use latest one in infer/activelearning tasks.  Any pre-conditions to publish?
        # shutil.copy(os.path.join(output_dir, "model.pt"), os.path.join(self.model_dir, "segmentation_heart.pt"))
        return task(max_epochs=epochs, amp=amp)
