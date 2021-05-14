import logging
import os

from lib import InferDeepgrow, InferSpleen, MyStrategy, TrainDeepgrow, TrainSpleen
from monai.networks.layers import Norm
from monai.networks.nets import BasicUNet, UNet

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")

        self.deepgrow_net = BasicUNet(dimensions=3, in_channels=3, out_channels=1, features=(32, 64, 128, 256, 512, 32))
        self.spleen_net = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

        infers = {
            "deepgrow": InferDeepgrow(os.path.join(self.model_dir, "deepgrow.pt"), network=self.deepgrow_net),
            "spleen": InferSpleen(os.path.join(self.model_dir, "spleen.pt"), network=self.spleen_net),
        }

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
        model = request.get("model", "deepgrow")
        name = request.get("name", "model_01")
        epochs = request.get("epochs", 1)
        amp = request.get("amp", True)
        device = request.get("device", "cuda")
        lr = request.get("lr", 0.0001)
        val_split = request.get("val_split", 0.2)

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        tasks = []
        if model == "deepgrow" or model == "all":
            output_dir = os.path.join(self.model_dir, f"deepgrow_{name}")

            load_path = os.path.join(output_dir, "model.pt")
            load_path = load_path if os.path.exists(load_path) else os.path.join(self.model_dir, "deepgrow.pt")
            logger.info(f"Using existing pre-trained weights: {load_path}")

            tasks.append(
                TrainDeepgrow(
                    roi_size=(128, 192, 192),
                    model_size=(128, 192, 192),
                    max_train_interactions=15,
                    max_val_interactions=20,
                    output_dir=output_dir,
                    data_list=self.datastore().datalist(),
                    network=self.deepgrow_net,
                    device=device,
                    lr=lr,
                    val_split=val_split,
                    load_path=load_path,
                )
            )
        if model == "spleen" or model == "all":
            output_dir = os.path.join(self.model_dir, f"spleen_{name}")

            load_path = os.path.join(output_dir, "model.pt")
            load_path = load_path if os.path.exists(load_path) else os.path.join(self.model_dir, "spleen.pt")
            logger.info(f"Using existing pre-trained weights: {load_path}")

            tasks.append(
                TrainSpleen(
                    output_dir=output_dir,
                    data_list=self.datastore().datalist(),
                    network=self.spleen_net,
                    device=device,
                    lr=lr,
                    val_split=val_split,
                    load_path=load_path,
                )
            )

        logger.info(f"Total Train tasks to run: {len(tasks)}")
        result = None
        for task in tasks:
            result = task(max_epochs=epochs, amp=amp)
        return result
