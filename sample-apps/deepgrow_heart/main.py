import json
import logging
import os

from lib import InferDeepgrow, MyStrategy, TrainDeepgrow
from monai.networks.nets import BasicUNet

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.infer.deepgrow_pipeline import InferDeepgrowPipeline

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")

        self.data = {
            "deepgrow_2d": {
                "network": BasicUNet(dimensions=2, in_channels=3, out_channels=1, features=(32, 64, 128, 256, 512, 32)),
                "path": [
                    os.path.join(self.model_dir, "deepgrow_2d.pt"),
                    os.path.join(self.model_dir, "deepgrow_2d_final.pt"),
                ],
                "url": "https://www.dropbox.com/s/t6kazwpvi2f1ppl/deepgrow_2d.pt?dl=1",
            },
            "deepgrow_3d": {
                "network": BasicUNet(dimensions=3, in_channels=3, out_channels=1, features=(32, 64, 128, 256, 512, 32)),
                "path": [
                    os.path.join(self.model_dir, "deepgrow_3d.pt"),
                    os.path.join(self.model_dir, "deepgrow_3d_final.pt"),
                ],
                "url": "https://www.dropbox.com/s/xgortm6ljd3dvhw/deepgrow_3d.pt?dl=1",
            },
            "segmentation_heart": {
                "network": None,
                "path": [os.path.join(self.model_dir, "segmentation_heart.ts")],
                "url": "https://www.dropbox.com/s/p60vfxpa4l64fmz/segmentation_spleen.ts?dl=1",
            },
        }

        deepgrow_3d = InferDeepgrow(
            path=self.data["deepgrow_3d"]["path"],
            network=self.data["deepgrow_3d"]["network"],
            dimension=3,
            model_size=(128, 192, 192),
        )
        infers = {
            "deepgrow": InferDeepgrowPipeline(
                path=self.data["deepgrow_2d"]["path"],
                network=self.data["deepgrow_2d"]["network"],
                model_3d=deepgrow_3d,
            ),
            "deepgrow_2d": InferDeepgrow(
                path=self.data["deepgrow_2d"]["path"],
                network=self.data["deepgrow_2d"]["network"],
                dimension=2,
                model_size=(256, 256),
            ),
            "deepgrow_3d": deepgrow_3d,
        }

        strategies = {
            "random": Random(),
            "first": MyStrategy(),
        }

        resources = [(self.data[k]["path"][0], self.data[k]["url"]) for k in self.data.keys()]

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            strategies=strategies,
            resources=resources,
        )

    def train(self, request):
        logger.info(f"Training request: {request}")
        model = request.get("model", "deepgrow_2d")

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        models = ["deepgrow_2d", "deepgrow_3d"] if model == "all" else [model]
        logger.info(f"Selected models for training: {models}")

        tasks = []
        for model in models:
            logger.info(f"Creating Training task for model: {model}")

            name = request.get("name", "model_01")
            output_dir = os.path.join(self.model_dir, f"{model}_{name}")
            data = self.data[model]

            network = data["network"]
            load_path = os.path.join(output_dir, "model.pt")
            load_path = load_path if os.path.exists(load_path) else data["path"][0]
            logger.info(f"Using existing pre-trained weights: {load_path}")

            # Datalist for train/validation
            train_d, val_d = self.partition_datalist(self.datastore().datalist(), request.get("val_split", 0.2))

            # Update/Publish latest model for infer/active learning use
            final_model = data["path"][1]
            train_stats_path = os.path.join(self.model_dir, f"train_stats_{model}.json")

            if model == "deepgrow_3d":
                task = TrainDeepgrow(
                    dimension=3,
                    roi_size=(128, 192, 192),
                    model_size=(128, 192, 192),
                    max_train_interactions=15,
                    max_val_interactions=20,
                    output_dir=output_dir,
                    train_datalist=train_d,
                    val_datalist=val_d,
                    network=network,
                    load_path=load_path,
                    publish_path=final_model,
                    stats_path=train_stats_path,
                    device=request.get("device", "cuda"),
                    lr=request.get("lr", 0.0001),
                )
            elif model == "deepgrow_2d":
                # TODO:: Flatten the dataset and batch it instead of picking random slice id
                task = TrainDeepgrow(
                    dimension=2,
                    roi_size=(256, 256),
                    model_size=(256, 256),
                    max_train_interactions=15,
                    max_val_interactions=5,
                    output_dir=output_dir,
                    train_datalist=train_d,
                    val_datalist=val_d,
                    network=network,
                    load_path=load_path,
                    publish_path=final_model,
                    stats_path=train_stats_path,
                    device=request.get("device", "cuda"),
                    lr=request.get("lr", 0.0001),
                )
            else:
                raise Exception(f"Train Definition for {model} Not Found")

            tasks.append(task)

        logger.info(f"Total Train tasks to run: {len(tasks)}")
        result = None
        for task in tasks:
            result = task(max_epochs=request.get("epochs", 1), amp=request.get("amp", True))
        return result

    def train_stats(self):
        # Return 3D stats as main stats and add 2D starts as part of
        res = {}
        for model in ["deepgrow_3d", "deepgrow_2d"]:
            train_stats_path = os.path.join(self.model_dir, f"train_stats_{model}.json")
            if os.path.exists(train_stats_path):
                with open(train_stats_path, "r") as fc:
                    if res:
                        res[model] = json.load(fc)
                    else:
                        res = json.load(fc)

        return res
