# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
from distutils.util import strtobool
from typing import Dict

from lib import DeepEdit, DeepEditSeg, MyStrategy, MyTrain
from monai.networks.nets import BasicUNet
from monai.networks.nets.dynunet import DynUNet

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.scribbles.infer import HistogramBasedGraphCut
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.activelearning.tta import TTA
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.tasks.scoring.tta import TTAScoring
from monailabel.utils.others.planner import HeuristicPlanner

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        network = conf.get("network", "dynunet")
        if network == "unet":
            network_params = {
                "dimensions": 3,
                "in_channels": 3,
                "out_channels": 1,
                "features": [32, 64, 128, 256, 512, 32],
            }
            self.network = BasicUNet(**network_params)
            self.network_with_dropout = BasicUNet(**network_params, dropout=0.2)
            self.find_unused_parameters = False
        else:
            network_params = {
                "spatial_dims": 3,
                "in_channels": 3,
                "out_channels": 1,
                "kernel_size": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "strides": [
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 1],
                ],
                "upsample_kernel_size": [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 1],
                ],
                "norm_name": "instance",
                "deep_supervision": False,
                "res_block": True,
            }
            self.network = DynUNet(**network_params)
            self.network_with_dropout = DynUNet(**network_params, dropout=0.2)
            self.find_unused_parameters = False

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, f"pretrained_{network}.pt")
        self.final_model = os.path.join(self.model_dir, f"model_{network}.pt")

        # Use Heuristic Planner to determine target spacing and spatial size based on dataset+gpu
        spatial_size = json.loads(conf.get("spatial_size", "[256, 256, 128]"))
        target_spacing = json.loads(conf.get("target_spacing", "[1.0, 1.0, 1.0]"))
        self.heuristic_planner = strtobool(conf.get("heuristic_planner", "false"))
        self.planner = HeuristicPlanner(spatial_size=spatial_size, target_spacing=target_spacing)

        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "true"))
        pretrained_model_uri = conf.get("pretrained_model_path", f"{self.PRE_TRAINED_PATH}deepedit_{network}_spleen.pt")

        # Path to pretrained weights
        if use_pretrained_model:
            logger.info(f"Pretrained Model Path: {pretrained_model_uri}")
            self.download([(self.pretrained_model, pretrained_model_uri)])

        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")

        self.tta_enabled = strtobool(conf.get("tta_enabled", "false"))
        self.tta_samples = int(conf.get("tta_samples", "5"))
        logger.info(f"TTA Enabled: {self.tta_enabled}; Samples: {self.tta_samples}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="DeepEdit",
            description="Active learning solution using DeepEdit to label 3D Images",
        )

    def init_datastore(self) -> Datastore:
        datastore = super().init_datastore()
        if self.heuristic_planner:
            self.planner.run(datastore)
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "deepedit": DeepEdit(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.planner.spatial_size,
                target_spacing=self.planner.target_spacing,
            ),
            "deepedit_seg": DeepEditSeg(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.planner.spatial_size,
                target_spacing=self.planner.target_spacing,
            ),
            # intensity range set for MRI
            "Histogram+GraphCut": HistogramBasedGraphCut(
                intensity_range=(0, 1500, 0.0, 1.0, True), pix_dim=(2.5, 2.5, 5.0), lamda=1.0, sigma=0.1
            ),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "deepedit_train": MyTrain(
                self.model_dir,
                self.network,
                spatial_size=self.planner.spatial_size,
                target_spacing=self.planner.target_spacing,
                load_path=self.pretrained_model,
                publish_path=self.final_model,
                config={"pretrained": strtobool(self.conf.get("use_pretrained_model", "true"))},
                debug_mode=strtobool(self.conf.get("debug", "false")),
                find_unused_parameters=self.find_unused_parameters,
            )
        }

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies["EPISTEMIC"] = Epistemic()
        if self.tta_enabled:
            strategies["TTA"] = TTA()
        strategies["random"] = Random()
        strategies["first"] = MyStrategy()
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        if self.epistemic_enabled:
            methods["EPISTEMIC"] = EpistemicScoring(
                model=[self.pretrained_model, self.final_model],
                network=self.network_with_dropout,
                transforms=self._infers["deepedit_seg"].pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        if self.tta_enabled:
            methods["TTA"] = TTAScoring(
                model=[self.pretrained_model, self.final_model],
                network=self.network,
                num_samples=self.tta_samples,
                spatial_size=self.planner.spatial_size,
                spacing=self.planner.target_spacing,
            )

        methods["dice"] = Dice()
        methods["sum"] = Sum()
        return methods


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", default="dynunet", choices=["unet", "dynunet"])
    parser.add_argument("-s", "--studies", default="/local/sachi/Datasets/Task09_Spleen/imagesTr")
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-d", "--dataset", default="CacheDataset")
    parser.add_argument("-o", "--output", default="model_01")
    parser.add_argument("-i", "--size", default="[256,256,128]")
    parser.add_argument("-b", "--batch", type=int, default=1)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "use_pretrained_model": "false",
        "auto_update_scoring": "false",
        "spatial_size": args.size,
        "network": args.network,
    }

    app = MyApp(app_dir, studies, conf)
    app.train(
        request={
            "name": args.output,
            "model": "deepedit_train",
            "max_epochs": args.epoch,
            "dataset": args.dataset,
            "train_batch_size": args.batch,
            "multi_gpu": True,
        }
    )


if __name__ == "__main__":
    main()
