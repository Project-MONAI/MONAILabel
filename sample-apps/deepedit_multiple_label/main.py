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

from lib import Deepgrow, MyTrain, Segmentation
from lib.activelearning import MyStrategy
from monai.networks.nets import DynUNet

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

        # background label is used to place the negative clicks
        # Zero values are reserved to background. Non zero values are for the labels
        self.label_names = {
            "spleen": 1,
            "right_kidney": 2,
            "left_kidney": 3,
            "liver": 6,
            "background": 0,
        }

        network_params = {
            "spatial_dims": 3,
            "in_channels": len(self.label_names) + 1,  # All labels plus Image
            "out_channels": len(self.label_names),  # All labels including background
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

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        # Use Heuristic Planner to determine target spacing and spatial size based on dataset+gpu
        spatial_size = json.loads(conf.get("spatial_size", "[128, 128, 128]"))
        target_spacing = json.loads(conf.get("target_spacing", "[1.5, 1.5, 2.0]"))
        self.heuristic_planner = strtobool(conf.get("heuristic_planner", "false"))
        self.planner = HeuristicPlanner(spatial_size=spatial_size, target_spacing=target_spacing)

        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "false"))
        pretrained_model_uri = conf.get("pretrained_model_path", f"{self.PRE_TRAINED_PATH}/deepedit_spleen.pt")

        # Path to pretrained weights
        if use_pretrained_model:
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
            "deepedit": Deepgrow(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.planner.spatial_size,
                target_spacing=self.planner.target_spacing,
            ),
            "deepedit_seg": Segmentation(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.planner.spatial_size,
                target_spacing=self.planner.target_spacing,
                label_names=self.label_names,
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
                label_names=self.label_names,
                debug_mode=False,
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app_dir_path = os.path.normpath("/home/adp20local/Documents/MONAILabel/sample-apps/deepedit_multiple_label")
    studies_path = os.path.normpath(
        "/home/adp20local/Documents/Datasets/monailabel_datasets/multilabel_abdomen/NIFTI/train"
    )
    # conf is Dict[str, str]
    conf = {
        "use_pretrained_model": "false",
        "heuristic_planner": "false",
        "tta_enabled": "false",
        "tta_samples": "10",
    }
    al_app = MyApp(app_dir=app_dir_path, studies=studies_path, conf=conf)
    request = {
        "device": "cuda",
        "model": "deepedit_train",
        "max_epochs": 300,
        "amp": False,
        "lr": 0.0001,
    }
    al_app.train(request=request)

    # # PERFORMING INFERENCE USING INTERACTIVE MODEL
    # deepgrow_3d = {
    #     "model": "deepedit",
    #     "image": f"{studies_path}/img0022.nii.gz",
    #     "foreground": {
    #         'spleen': [[[61, 106, 54], [65, 106, 54]]],
    #         'right_kidney': [[[59, 86, 93]]],
    #         'left_kidney': [[[61, 94, 54]]],
    #         'liver': [[[79, 104, 67], [94, 114, 67]]],
    #     },
    #     "background": [[[6, 132, 427]]],
    # }
    # al_app.infer(deepgrow_3d)

    # # PERFORMING INFERENCE USING AUTOMATIC MODEL
    # automatic_request = {
    #     "model": "deepedit_seg",
    #     "image": f"{studies_path}/img0022.nii.gz",
    # }
    # al_app.infer(automatic_request)
    #
    # return None


if __name__ == "__main__":
    main()
