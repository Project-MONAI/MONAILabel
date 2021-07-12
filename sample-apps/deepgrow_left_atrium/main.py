import copy
import json
import logging
import os

from lib import InferDeepgrow, MyStrategy, TrainDeepgrow
from monai.apps import load_from_mmar

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.infer.deepgrow_pipeline import InferDeepgrowPipeline

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")

        self.model_dir_2d = os.path.join(self.model_dir, "deepgrow_2d")
        self.pretrained_model_2d = os.path.join(self.model_dir, "deepgrow_2d", "pretrained.pt")
        self.final_model_2d = os.path.join(self.model_dir, "deepgrow_2d", "model.pt")
        self.train_stats_path_2d = os.path.join(self.model_dir, "deepgrow_2d", "train_stats.json")
        self.mmar_2d = "clara_pt_deepgrow_2d_annotation_1"

        self.model_dir_3d = os.path.join(self.model_dir, "deepgrow_3d")
        self.pretrained_model_3d = os.path.join(self.model_dir, "deepgrow_3d", "pretrained.pt")
        self.final_model_3d = os.path.join(self.model_dir, "deepgrow_3d", "model.pt")
        self.train_stats_path_3d = os.path.join(self.model_dir, "deepgrow_3d", "train_stats.json")
        self.mmar_3d = "clara_pt_deepgrow_3d_annotation_1"

        self.download(
            [
                (
                    self.pretrained_model_2d,
                    "https://github.com/Project-MONAI/MONAILabel/releases/download/data/deepgrow_2d_left_atrium.pt",
                ),
                (
                    self.pretrained_model_3d,
                    "https://github.com/Project-MONAI/MONAILabel/releases/download/data/deepgrow_3d_left_atrium.pt",
                ),
            ]
        )

        super().__init__(app_dir, studies)

    def init_infers(self):
        infers = {
            "deepgrow_2d": InferDeepgrow(
                [self.pretrained_model_2d, self.final_model_2d],
                load_from_mmar(self.mmar_2d, self.model_dir_2d, pretrained=False),
            ),
            "deepgrow_3d": InferDeepgrow(
                [self.pretrained_model_3d, self.final_model_3d],
                load_from_mmar(self.mmar_3d, self.model_dir_3d, pretrained=False),
                dimension=3,
                model_size=(128, 192, 192),
            ),
        }

        infers["deepgrow_pipeline"] = InferDeepgrowPipeline(
            path=[self.pretrained_model_2d, self.final_model_2d],
            network=load_from_mmar(self.mmar_2d, self.model_dir_2d, pretrained=False),
            model_3d=infers["deepgrow_3d"],
            description="Combines Deepgrow 2D model and 3D deepgrow model",
        )
        return infers

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }

    def train(self, request):
        logger.info(f"Training request: {request}")

        model = request.get("model", "deepgrow_2d")
        models = ["deepgrow_2d", "deepgrow_3d"] if model == "all" else [model]
        logger.info(f"Selected models for training: {models}")

        tasks = []
        for model in models:
            logger.info(f"Creating Training task for model: {model}")

            if model == "deepgrow_2d":
                mmar = self.mmar_2d
                model_dir = self.model_dir_2d
                final_model = self.final_model_2d
                train_stats_path = self.train_stats_path_2d
            else:
                mmar = self.mmar_3d
                model_dir = self.model_dir_3d
                final_model = self.final_model_3d
                train_stats_path = self.train_stats_path_3d

            output_dir = os.path.join(model_dir, request.get("name", "model_01"))

            # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
            load_path = os.path.join(output_dir, "model.pt")
            if not os.path.exists(load_path) and request.get("pretrained", True):
                load_path = None
                network = load_from_mmar(mmar, model_dir)
            else:
                network = load_from_mmar(mmar, model_dir, pretrained=False)

            # Datalist for train/validation
            train_d, val_d = self.partition_datalist(self.datastore().datalist(), request.get("val_split", 0.2))

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
            train_stats_path = os.path.join(self.model_dir, model, "train_stats.json")
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
