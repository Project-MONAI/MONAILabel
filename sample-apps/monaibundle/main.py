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
import re
import shutil
import sys
from typing import Dict

import requests
from monai.bundle import download

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.generic import MONAI_ZOO_INFO, MONAI_ZOO_REPO, MONAI_ZOO_SOURCE, strtobool

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        zoo_info = requests.get(conf.get("zoo_info", MONAI_ZOO_INFO)).json()
        zoo_source = conf.get("zoo_source", MONAI_ZOO_SOURCE)
        zoo_repo = conf.get("zoo_repo", MONAI_ZOO_REPO)

        available = {k.replace(".zip", ""): v for k, v in zoo_info.items()}
        models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) separated names to pass multiple")
            print("    -c models {}".format("\n    -c models ".join(available.keys())))
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if not available.get(m) and not os.path.isdir(os.path.join(self.model_dir, m))]

        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) separated names to pass multiple")
            print("    -c models {}".format("\n    -c models ".join(available.keys())))
            print(f"Or provide valid local bundle directories under: {self.model_dir}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)
        self.models: Dict[str, str] = {}

        for k in models:
            v = available.get(k)
            p = os.path.join(self.model_dir, k)
            if not v:
                logger.info(f"+++ Adding Bundle from Local: {k} => {p}")
            else:
                logger.info(f"+++ Adding Bundle from Zoo: {k} => {v} => {p}")
                if not os.path.exists(p):
                    download(name=k, bundle_dir=self.model_dir, source=zoo_source, repo=zoo_repo)
                    e = os.path.join(self.model_dir, re.sub(r"_v.*.zip", "", f"{k}.zip"))
                    if os.path.isdir(e):
                        shutil.move(e, p)
            sys.path.append(p)

            self.models[k] = p

        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Zoo/Bundle ({monailabel.__version__})",
            description="DeepLearning models provided via MONAI Zoo/Bundle",
            version=monailabel.__version__,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################
        for n, b in self.models.items():
            i = BundleInferTask(b, self.conf)
            logger.info(f"+++ Adding Inferer:: {n} => {i}")
            infers[n] = i
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers

        for n, b in self.models.items():
            t = BundleTrainTask(b, self.conf)
            if not t or not t.is_valid():
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t
        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
            "first": First(),
        }

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse
    from pathlib import Path

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.png", "*.jpg", "*.jpeg", ".nii", ".nii.gz"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Datasets/Radiology"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies

    app = MyApp(app_dir, studies, {"preload": "true", "models": "spleen_ct_segmentation_v0.1.0"})
    train(app)


def infer(app):
    import json
    import shutil

    res = app.infer(
        request={
            "model": "spleen_ct_segmentation_v0.1.0",
            "image": "image",
        }
    )

    print(json.dumps(res, indent=2))
    shutil.move(res["label"], os.path.join(app.studies, "test"))
    logger.info("All Done!")


def train(app):
    app.train(
        request={
            "model": "spleen_ct_segmentation_v0.1.0",
            "max_epochs": 2,
            "multi_gpu": False,
            "val_split": 0.1,
            "val_interval": 1,
        },
    )


if __name__ == "__main__":
    main()
