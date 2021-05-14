import logging
import os

from lib import MyInfer, MyStrategy, MyTrain
from monai.networks.nets import BasicUNet

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")

        self.network = BasicUNet(dimensions=3, in_channels=3, out_channels=1, features=(32, 64, 128, 256, 512, 32))

        infers = {"deepgrow": MyInfer(os.path.join(self.model_dir, "deepgrow.pt"), network=self.network)}

        strategies = {
            "random": Random(),
            "first": MyStrategy(),
        }

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            strategies=strategies,
        )

    def train(self, request):
        logger.info(f"Training request: {request}")
        name = request.get("name", "model_01")
        epochs = request.get("epochs", 1)
        amp = request.get("amp", True)
        device = request.get("device", "cuda")
        lr = request.get("lr", 0.0001)
        val_split = request.get("val_split", 0.2)

        output_dir = os.path.join(self.model_dir, name)

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        load_path = load_path if os.path.exists(load_path) else os.path.join(self.model_dir, "deepgrow.pt")
        logger.info(f"Using existing pre-trained weights: {load_path}")

        task = MyTrain(
            roi_size=(128, 192, 192),
            model_size=(128, 192, 192),
            max_train_interactions=15,
            max_val_interactions=20,
            output_dir=output_dir,
            data_list=self.datastore().datalist(),
            network=self.network,
            device=device,
            lr=lr,
            val_split=val_split,
            load_path=load_path,
        )

        return task(max_epochs=epochs, amp=amp)
