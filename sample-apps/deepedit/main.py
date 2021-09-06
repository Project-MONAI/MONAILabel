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

from lib import Deepgrow, MyStrategy, MyTrain, Segmentation
from monai.networks.nets.dynunet_v1 import DynUNetV1

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.others.planner import ExperimentPlanner

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):

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
        self.spatial_size = None
        self.target_spacing = None

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="DeepEdit",
            description="Active learning solution using DeepEdit to label 3D Images",
            version=2,
        )

    def experiment_planner(self):
        # Experiment planner
        self.planner = ExperimentPlanner(datastore=self.datastore())
        self.spatial_size = self.planner.get_target_img_size()
        self.target_spacing = self.planner.get_target_spacing()
        logger.info(f"Available GPU memory: {list(self.planner.get_gpu_memory_map().values())} in MB")

    def init_infers(self):
        self.experiment_planner()
        logger.info(f"Spacing set: {self.target_spacing}")
        logger.info(f"Spatial size set: {self.spatial_size}")
        return {
            "deepedit": Deepgrow(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
            ),
            "automatic": Segmentation(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
            ),
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
                config={"pretrained": False},
                debug_mode=False,
            )
        }

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }
