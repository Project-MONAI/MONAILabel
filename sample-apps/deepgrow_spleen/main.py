import copy
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
                "url": "https://www.dropbox.com/s/u9ka8l3kxr8m5ys/deepgrow_2d_spleen.pt?dl=1",
            },
            "deepgrow_3d": {
                "network": BasicUNet(dimensions=3, in_channels=3, out_channels=1, features=(32, 64, 128, 256, 512, 32)),
                "path": [
                    os.path.join(self.model_dir, "deepgrow_3d.pt"),
                    os.path.join(self.model_dir, "deepgrow_3d_final.pt"),
                ],
                "url": "https://www.dropbox.com/s/6krff7zjk3nkk16/deepgrow_3d_spleen.pt?dl=1",
            },
        }

        deepgrow_3d = InferDeepgrow(
            path=self.data["deepgrow_3d"]["path"],
            network=self.data["deepgrow_3d"]["network"],
            dimension=3,
            model_size=(96, 96, 96),
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
                    roi_size=(96, 96, 96),
                    model_size=(96, 96, 96),
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
                    max_epochs=request.get("epochs", 1),
                    amp=request.get("amp", True),
                    train_batch_size=request.get("train_batch_size", 1),
                    val_batch_size=request.get("val_batch_size", 1),
                )
            elif model == "deepgrow_2d":
                flatten_train_d = []
                for _ in range(max(request.get("2d_train_random_slices", 20), 1)):
                    flatten_train_d.extend(copy.deepcopy(train_d))
                logger.info(f"After flatten:: {len(train_d)} => {len(flatten_train_d)}")

                flatten_val_d = []
                for _ in range(max(request.get("2d_val_random_slices", 5), 1)):
                    flatten_val_d.extend(copy.deepcopy(val_d))
                logger.info(f"After flatten:: {len(val_d)} => {len(flatten_val_d)}")

                task = TrainDeepgrow(
                    dimension=2,
                    roi_size=(256, 256),
                    model_size=(256, 256),
                    max_train_interactions=15,
                    max_val_interactions=5,
                    output_dir=output_dir,
                    train_datalist=flatten_train_d,
                    val_datalist=flatten_val_d,
                    network=network,
                    load_path=load_path,
                    publish_path=final_model,
                    stats_path=train_stats_path,
                    device=request.get("device", "cuda"),
                    lr=request.get("lr", 0.0001),
                    max_epochs=request.get("2d_epochs", 1),
                    amp=request.get("amp", True),
                    train_batch_size=request.get("2d_train_batch_size", 4),
                    val_batch_size=request.get("2d_val_batch_size", 4),
                )
            else:
                raise Exception(f"Train Definition for {model} Not Found")

            tasks.append(task)

        logger.info(f"Total Train tasks to run: {len(tasks)}")
        result = None
        for task in tasks:
            result = task()
        return result

    def train_stats(self):
        # Return both 2D and 3D stats.  Set current running or deepgrow_3d stats as active
        res = {}
        active = {}
        start_ts = 0
        for model in ["deepgrow_3d", "deepgrow_2d"]:
            train_stats_path = os.path.join(self.model_dir, f"train_stats_{model}.json")
            if os.path.exists(train_stats_path):
                with open(train_stats_path, "r") as fc:
                    r = json.load(fc)
                    res[model] = r

                    # Set current running or last ran model as active
                    if not active or r.get("current_time") or r.get("start_ts", 0) > start_ts:
                        start_ts = r.get("start_ts", 0)
                        active = copy.deepcopy(r)

        active.update(res)
        return active
