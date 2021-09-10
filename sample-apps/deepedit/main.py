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

from lib import Deepgrow, MyTrain, Segmentation
from lib.activelearning import MyStrategy
from monai.networks.nets.dynunet_v1 import DynUNetV1

from monailabel.interfaces.app import MONAILabelApp
from monailabel.scribbles.infer import HistogramBasedGraphCut
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.activelearning.tta import TTAStrategy
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.sum import Sum
from monailabel.tasks.scoring.tta import TTAScoring
from monailabel.utils.others.planner import ExperimentPlanner

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.network = DynUNetV1(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            kernel_size=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            upsample_kernel_size=[
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        # Use Experiment Planner to determine target spacing and spatial size based on dataset+gpu
        self.use_experiment_planner = strtobool(conf.get("use_experiment_planner", "false"))
        self.spatial_size = json.loads(conf.get("spatial_size", "[128, 128, 64]"))
        self.target_spacing = json.loads(conf.get("target_spacing", "[1.0, 1.0, 1.0]"))

        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "true"))
        pretrained_model_uri = conf.get("pretrained_model_path", f"{self.PRE_TRAINED_PATH}/deepedit_left_atrium.pt")

        # Path to pretrained weights
        if use_pretrained_model:
            self.download([(self.pretrained_model, pretrained_model_uri)])

        self.tta_enabled = strtobool(conf.get("tta_enabled", "true"))
        self.tta_samples = int(conf.get("tta_samples", "5"))
        logger.info(f"TTA Enabled: {self.tta_enabled}; Samples: {self.tta_samples}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="DeepEdit",
            description="Active learning solution using DeepEdit to label 3D Images",
        )

    def _init_planner(self):
        # Experiment planner to compute spatial_size and target spacing based on images/gpu resources
        planner = ExperimentPlanner(datastore=self.datastore())
        self.spatial_size = planner.get_target_img_size()
        self.target_spacing = planner.get_target_spacing()

    def init_infers(self):
        if self.use_experiment_planner:
            self._init_planner()
        logger.info(f"Using Spacing: {self.target_spacing}; Spatial Size: {self.spatial_size}")

        return {
            "deepedit": Deepgrow(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
            ),
            "deepedit_seg": Segmentation(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
            ),
            "histogramBasedGraphCut": HistogramBasedGraphCut(),
        }

    def init_trainers(self):
        return {
            "deepedit_train": MyTrain(
                self.model_dir,
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                load_path=self.pretrained_model,
                publish_path=self.final_model,
                config={"pretrained": strtobool(self.conf.get("use_pretrained_model", "true"))},
                debug_mode=False,
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
                model=[self.pretrained_model, self.final_model], network=self.network, num_samples=self.tta_samples
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
