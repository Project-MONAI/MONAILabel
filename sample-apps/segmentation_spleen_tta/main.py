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

from lib import MyInfer, MyTrain
from lib.activelearning import TTA, MyStrategy
from monai.networks.nets.dynunet_v1 import DynUNetV1

# from monailabel.endpoints.utils import BackgroundTask
from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.scoring.tta_scoring import TTAScoring

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):

        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.network = DynUNetV1(
            spatial_dims=3,
            in_channels=1,
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

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Segmentation - Spleen",
            description="Active Learning solution to label Spleen Organ over 3D CT Images",
            version=2,
        )

        # TO DO - start scoring after starting App if trained model exists??
        # BackgroundTask.run("scoring", request={"method": "TTA"}, params={}, force_sync=False)

    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, self.network),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_trainers(self):
        config = {
            "name": "model_01",
            "pretrained": False,
            "device": "cuda",
            "max_epochs": 200,
            "val_split": 0.2,
            "train_batch_size": 1,
            "val_batch_size": 1,
        }
        return {
            "segmentation_spleen": MyTrain(
                self.model_dir,
                self.network,
                publish_path=self.final_model,
                config=config,
            )
        }

    def init_strategies(self):
        return {
            "TTA": TTA(),
            "random": Random(),
            "first": MyStrategy(),
        }

    def init_scoring_methods(self):
        return {
            "TTA": TTAScoring(model=self.init_infers()["segmentation_spleen"], plot=False),
        }
