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

from lib import MyInfer, MyStrategy, MyTrain
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monailabel.interfaces.app import MONAILabelApp
from monailabel.scribbles.infer import HistogramBasedGraphCut
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.activelearning.tta import TTAStrategy
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.sum import Sum
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
        ngc_path = "https://api.ngc.nvidia.com/v2/models/nvidia/med/"
        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "true"))
        pretrained_model_uri = conf.get(
            "pretrained_model_path", f"{ngc_path}/clara_pt_spleen_ct_segmentation/versions/1/files/models/model.pt"
        )
        if use_pretrained_model:
            self.download([(self.pretrained_model, pretrained_model_uri)])

        self.tta_enabled = strtobool(conf.get("tta_enabled", "false"))
        self.tta_samples = int(conf.get("tta_samples", "5"))
        logger.info(f"TTA Enabled: {self.tta_enabled}; Samples: {self.tta_samples}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Segmentation - Generic",
            description="Active Learning solution to label generic organ",
        )

    def init_infers(self):
        infers = {
            "segmentation": MyInfer([self.pretrained_model, self.final_model], self.network),
            "histogramBasedGraphCut": HistogramBasedGraphCut(),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_trainers(self):
        return {
            "segmentation": MyTrain(
                self.model_dir, self.network, load_path=self.pretrained_model, publish_path=self.final_model
            )
        }

    def init_strategies(self):
        return {
            "TTA": TTAStrategy(),
            "random": Random(),
            "first": MyStrategy(),
        }

    def init_scoring_methods(self):
        return {
            "TTA": TTAScoring(
                model=[self.pretrained_model, self.final_model],
                network=self.network,
                deepedit=False,
                num_samples=self.tta_samples,
            ),
            "sum": Sum(),
            "dice": Dice(),
        }

    def on_init_complete(self):
        super().on_init_complete()
        self._run_tta_scoring()

    def next_sample(self, request):
        res = super().next_sample(request)
        self._run_tta_scoring()
        return res

    def train(self, request):
        res = super().train(request)
        self._run_tta_scoring()
        return res

    def _run_tta_scoring(self):
        if self.tta_enabled:
            self.async_scoring("TTA")
