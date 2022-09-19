# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from datetime import timedelta
from typing import Dict

import lib.configs
import schedule
from timeloop import Timeloop

import monailabel
from monailabel.config import settings
from monailabel.datastore.cvat import CVATDatastore
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.random import Random
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import create_dataset_from_path, strtobool

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}

        models = conf.get("models", "all")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, None)

        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Endoscopy ({monailabel.__version__})",
            description="DeepLearning models for endoscopy",
            version=monailabel.__version__,
        )
        self.downloading = False

    def init_datastore(self) -> Datastore:
        if settings.MONAI_LABEL_DATASTORE_URL and settings.MONAI_LABEL_DATASTORE.lower() == "cvat":
            logger.info(f"Using CVAT: {self.studies}")
            return CVATDatastore(
                datastore_path=self.studies,
                api_url=settings.MONAI_LABEL_DATASTORE_URL,
                username=settings.MONAI_LABEL_DATASTORE_USERNAME,
                password=settings.MONAI_LABEL_DATASTORE_PASSWORD,
                project=self.conf.get("cvat_project", "MONAILabel"),
                task_prefix=self.conf.get("cvat_task_prefix", "ActiveLearning_Iteration"),
                image_quality=int(self.conf.get("cvat_image_quality", "70")),
                labels=self.conf.get("cvat_labels"),
                normalize_label=strtobool(self.conf.get("cvat_normalize_label", "true")),
                segment_size=int(self.conf.get("cvat_segment_size", "1")),
                extensions=settings.MONAI_LABEL_DATASTORE_FILE_EXT,
                auto_reload=settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD,
            )

        return super().init_datastore()

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers

        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t
        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
        }

        if strtobool(self.conf.get("skip_strategies", "false")):
            return strategies

        for n, task_config in self.models.items():
            s = task_config.strategy()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Strategy:: {k} => {v}")
                strategies[k] = v

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        if strtobool(self.conf.get("skip_scoring", "false")):
            return methods

        for n, task_config in self.models.items():
            s = task_config.scoring_method()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Scoring Method:: {k} => {v}")
                methods[k] = v

        logger.info(f"Active Learning Scoring Methods:: {list(methods.keys())}")
        return methods

    def on_init_complete(self):
        super().on_init_complete()
        if not isinstance(self.datastore(), CVATDatastore):
            return

        # Check for CVAT Task if complete and trigger training
        def update_model():
            if self.downloading:
                return

            try:
                self.downloading = True
                ds = self.datastore()
                if isinstance(ds, CVATDatastore):
                    name = ds.download_from_cvat()
                    if name:
                        models = self.conf.get("auto_finetune_models")
                        models = models.split(",") if models else models
                        logger.info(f"Trigger Training for model(s): {models}; Iteration Name: {name}")
                        self.async_training(model=models, params={"name": name})
                else:
                    logger.info("Nothing to update;  No new labels downloaded/refreshed from CVAT")
            finally:
                self.downloading = False

        time_loop = Timeloop()
        interval_in_sec = int(self.conf.get("auto_finetune_check_interval", "60"))
        schedule.every(interval_in_sec).seconds.do(update_model)

        @time_loop.job(interval=timedelta(seconds=interval_in_sec))
        def run_scheduler():
            schedule.run_pending()

        time_loop.start(block=False)


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse
    from pathlib import Path

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.png", "*.jpg", "*.jpeg", ".xml"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Dataset/Holoscan/tiny/images"
    # studies = f"{home}/Dataset/picked/all"
    # studies = f"{home}/Dataset/Holoscan/flattened/images"
    # studies = f"{home}/Dataset/Holoscan/tiny_flat/images"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies

    app = MyApp(app_dir, studies, {"preload": "true", "models": "deepedit"})
    logger.info(app.datastore().status())
    for _ in range(3):
        infer_deepedit(app)


def randamize_ds(train_datalist, val_datalist):
    import random

    half_train = len(train_datalist) // 2
    t1 = train_datalist[:half_train]
    t2 = train_datalist[half_train:]
    random.shuffle(t1)
    random.shuffle(t2)
    train_datalist = t1 + t2

    half_val = len(val_datalist) // 2
    v1 = val_datalist[:half_val]
    v2 = val_datalist[half_val:]
    random.shuffle(v1)
    random.shuffle(v2)
    val_datalist = v1 + v2

    return train_datalist, val_datalist


def deepedit_partition_datalist():
    train_datalist = create_dataset_from_path("/localhome/sachi/Dataset/Holoscan/105162/train", lab_ext=".jpg")
    val_datalist = create_dataset_from_path("/localhome/sachi/Dataset/Holoscan/105162/valid", lab_ext=".jpg")
    logger.info(f"******* Total Training   Dataset: {len(train_datalist)}")
    logger.info(f"******* Total Validation Dataset: {len(val_datalist)}")

    train_datalist = train_datalist[:100]
    val_datalist = val_datalist[:20]
    # train_datalist = train_datalist[:3200]
    # val_datalist = val_datalist[:400]

    train_datalist, val_datalist = randamize_ds(train_datalist, val_datalist)

    logger.info(f"******* Total Training   Dataset: {len(train_datalist)}")
    logger.info(f"******* Total Validation Dataset: {len(val_datalist)}")
    return train_datalist, val_datalist


def train_deepedit(app):
    import json

    train_ds, val_ds = deepedit_partition_datalist()
    train_ds_json = "/localhome/sachi/Dataset/Holoscan/deepedit_train_ds.json"
    val_ds_json = "/localhome/sachi/Dataset/Holoscan/deepedit_val_ds.json"

    with open(train_ds_json, "w") as fp:
        json.dump(train_ds, fp, indent=2)
    with open(val_ds_json, "w") as fp:
        json.dump(val_ds, fp, indent=2)

    res = app.train(
        request={
            "model": "deepedit",
            "max_epochs": 10,
            "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 10,
            "val_batch_size": 10,
            "multi_gpu": True,
            "val_split": 0.15,
            "pretrained": True,
            "train_ds": train_ds_json,
            "val_ds": val_ds_json,
        }
    )
    print(res)
    logger.info("All Done!")


def infer_deepedit(app):
    import shutil
    from pathlib import Path

    # import numpy as np
    # from PIL import Image
    # image = np.array(Image.open(os.path.join(app.studies, "Video_8_2020_01_13_Video2_Trim_01-25_f10200.jpg")))
    image = "Video_8_2020_01_13_Video2_Trim_01-25_f10200"

    res = app.infer(
        request={
            "model": "deepedit",
            "image": image,
            "foreground": [],
            "background": [],
            "output": "asap",
            # "result_extension": ".png",
        }
    )

    # print(json.dumps(res, indent=2))
    home = str(Path.home())
    shutil.move(res["label"], f"{home}/Dataset/output_image.xml")
    logger.info("All Done!")


def train_tooltracking(app):
    res = app.train(
        request={
            "model": "tooltracking",
            "max_epochs": 10,
            "dataset": "Dataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 4,
            "val_batch_size": 2,
            "multi_gpu": False,
            "val_split": 0.1,
        }
    )
    print(res)
    logger.info("All Done!")


def infer_tooltracking(app):
    import shutil
    from pathlib import Path

    res = app.infer(
        request={
            "model": "tooltracking",
            "image": "Video_8_2020_01_13_Video2_Trim_01-25_f10200",
            "output": "asap",
            # 'result_extension': '.png',
        }
    )

    # print(json.dumps(res, indent=2))
    home = str(Path.home())
    shutil.move(res["label"], f"{home}/Dataset/output_image.xml")
    logger.info("All Done!")


def infer_inbody(app):
    import json

    res = app.infer(
        request={
            "model": "inbody",
            "image": "Video_8_2020_01_13_Video2_Trim_01-25_f10200",
            # "logging": "ERROR",
        }
    )

    print(json.dumps(res["params"]["prediction"]))


def train_inbody(app):
    res = app.train(
        request={
            "model": "inbody",
            "max_epochs": 10,
            "dataset": "Dataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 1,
            "val_batch_size": 1,
            "multi_gpu": False,
            "val_split": 0.1,
        }
    )
    print(res)
    logger.info("All Done!")


if __name__ == "__main__":
    main()
