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

import logging
import os
from distutils.util import strtobool
from typing import Dict

from lib import MyInfer, MyStrategy, MyTrain
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.scribbles.infer import HistogramBasedGraphCut
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.activelearning.tta import TTA
from monailabel.tasks.scoring.tta import TTAScoring

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.network = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        # Path to pretrained weights
        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "true"))
        pretrained_model_uri = conf.get("pretrained_model_path", f"{self.PRE_TRAINED_PATH}/segmentation_left_atrium.pt")
        if use_pretrained_model:
            self.download([(self.pretrained_model, pretrained_model_uri)])

        self.tta_enabled = strtobool(conf.get("tta_enabled", "false"))
        self.tta_samples = int(conf.get("tta_samples", "5"))
        logger.info(f"TTA Enabled: {self.tta_enabled}; Samples: {self.tta_samples}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Segmentation - Left Atrium",
            description="Active Learning solution to label left atrium over 3D MRI Images",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "segmentation_left_atrium": MyInfer([self.pretrained_model, self.final_model], self.network),
            "Histogram+GraphCut": HistogramBasedGraphCut(),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "segmentation_left_atrium": MyTrain(
                self.model_dir, self.network, load_path=self.pretrained_model, publish_path=self.final_model
            )
        }

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {}
        if self.tta_enabled:
            strategies["TTA"] = TTA()
        strategies["random"] = Random()
        strategies["first"] = MyStrategy()
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        if self.tta_enabled:
            methods["TTA"] = TTAScoring(
                model=[self.pretrained_model, self.final_model],
                network=self.network,
                num_samples=self.tta_samples,
                spatial_size=(128, 128, 64),
                spacing=(1.0, 1.0, 1.0),
            )
        return methods
