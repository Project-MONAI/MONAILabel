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
from typing import Dict

from monai.transforms import Invertd, SaveImaged

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.scoring.epistemic_v2 import EpistemicScoring
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.generic import get_bundle_models, strtobool

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.models = get_bundle_models(app_dir, conf)
        # Add Epistemic model for scoring
        self.epistemic_models = (
            get_bundle_models(app_dir, conf, conf_key="epistemic_model") if conf.get("epistemic_model") else None
        )
        if self.epistemic_models:
            # Get epistemic parameters
            self.epistemic_max_samples = int(conf.get("epistemic_max_samples", "0"))
            self.epistemic_simulation_size = int(conf.get("epistemic_simulation_size", "5"))
            self.epistemic_dropout = float(conf.get("epistemic_dropout", "0.2"))

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
            if "deepedit" in n:
                # Adding automatic inferer
                i = BundleInferTask(b, self.conf, type="segmentation")
                logger.info(f"+++ Adding Inferer:: {n}_seg => {i}")
                infers[n + "_seg"] = i
                # Adding inferer for managing clicks
                i = BundleInferTask(b, self.conf, type="deepedit", deepedit=True)
                logger.info("+++ Adding DeepEdit Inferer")
                infers[n] = i
            else:
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

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        if not self.conf.get("epistemic_model"):
            return methods

        for n, b in self.epistemic_models.items():
            # Create BundleInferTask task with dropout instantiation for scoring inference
            i = BundleInferTask(
                b,
                self.conf,
                train_mode=True,
                skip_writer=True,
                dropout=self.epistemic_dropout,
                post_filter=[SaveImaged, Invertd],
            )
            methods[n] = EpistemicScoring(
                i, max_samples=self.epistemic_max_samples, simulation_size=self.epistemic_simulation_size
            )
            if not methods:
                continue
            methods = methods if isinstance(methods, dict) else {n: methods[n]}
            logger.info(f"+++ Adding Scoring Method:: {n} => {b}")

        logger.info(f"Active Learning Scoring Methods:: {list(methods.keys())}")
        return methods


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
    # studies = f"{home}/Datasets/Radiology"
    studies = "/tmp/testLiver/imagesTs/"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies

    app = MyApp(app_dir, studies, {"preload": "false", "models": "spleen_deepedit_annotation"})
    # train(app)
    infer(app)


def infer(app):
    import json
    import shutil

    res = app.infer(
        request={
            "model": "spleen_deepedit_annotation",
            "image": "image",
        }
    )

    print(json.dumps(res, indent=2))
    shutil.move(res["label"], os.path.join(app.studies, "test"))
    logger.info("All Done!")


def train(app):
    app.train(
        request={
            "model": "spleen_deepedit_annotation",
            "max_epochs": 2,
            "multi_gpu": False,
            "val_split": 0.1,
            "val_interval": 1,
        },
    )


if __name__ == "__main__":
    main()
