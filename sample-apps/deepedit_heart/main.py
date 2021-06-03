import logging
import os

from lib import Deepgrow, MyStrategy, MyTrain, Segmentation
from monai.networks.nets import DynUNet

from monailabel.interfaces import MONAILabelApp
from monailabel.interfaces.tasks import BatchInferImageType
from monailabel.utils.activelearning import TTA, Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.network = DynUNet(
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

        self.pretrained_model = os.path.join(self.model_dir, "deepedit.pt")
        self.final_model = os.path.join(self.model_dir, "final.pt")
        path = [self.pretrained_model, self.final_model]

        infers = {
            "deepedit": Deepgrow(path, self.network),
            "heart": Segmentation(path, self.network),
        }

        strategies = {
            "random": Random(),
            "first": MyStrategy(),
            "tta": TTA(path, self.network),
        }

        resources = [
            (self.pretrained_model, "https://www.dropbox.com/s/fgygtodqztjjdyz/deepedit_heart.pt?dl=1"),
        ]

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            strategies=strategies,
            resources=resources,
        )

    def train(self, request):
        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, request.get("name", "model_01"))

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        load_path = load_path if os.path.exists(load_path) else self.pretrained_model

        task = MyTrain(
            output_dir=output_dir,
            data_list=self.datastore().datalist(),
            network=self.network,
            load_path=load_path,
            publish_path=self.final_model,
            device=request.get("device", "cuda"),
            lr=request.get("lr", 0.0001),
            val_split=request.get("val_split", 0.2),
        )

        result = task(max_epochs=request.get("epochs", 1), amp=request.get("amp", True))

        # Compute Dice for new model over submitted/final labels
        self.batch_infer({"model": "heart", "images": BatchInferImageType.IMAGES_LABELED})
        self.scoring({"method": "dice"})
        return result
