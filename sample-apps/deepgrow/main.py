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

from lib import InferDeepgrow, MyStrategy, TrainDeepgrow
from monai.apps import load_from_mmar

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.infer.deepgrow_pipeline import InferDeepgrowPipeline

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")

        self.model_dir_2d = os.path.join(self.model_dir, "deepgrow_2d")
        self.final_model_2d = os.path.join(self.model_dir, "deepgrow_2d", "model.pt")
        self.train_stats_path_2d = os.path.join(self.model_dir, "deepgrow_2d", "train_stats.json")
        self.mmar_2d = "clara_pt_deepgrow_2d_annotation_1"

        self.model_dir_3d = os.path.join(self.model_dir, "deepgrow_3d")
        self.final_model_3d = os.path.join(self.model_dir, "deepgrow_3d", "model.pt")
        self.train_stats_path_3d = os.path.join(self.model_dir, "deepgrow_3d", "train_stats.json")
        self.mmar_3d = "clara_pt_deepgrow_3d_annotation_1"

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Deepgrow - Generic",
            description="Active Learning solution to label generic organ",
            version=2,
            labels=["spleen"],
        )

    def init_infers(self):
        infers = {
            "deepgrow_2d": InferDeepgrow(self.final_model_2d, load_from_mmar(self.mmar_2d, self.model_dir_2d)),
            "deepgrow_3d": InferDeepgrow(
                self.final_model_3d,
                load_from_mmar(self.mmar_3d, self.model_dir_3d),
                dimension=3,
                model_size=(128, 192, 192),
            ),
        }

        infers["deepgrow_pipeline"] = InferDeepgrowPipeline(
            path=None,
            network=load_from_mmar(self.mmar_2d, self.model_dir_2d),
            model_3d=infers["deepgrow_3d"],
            description="Combines Deepgrow 2D model and 3D deepgrow model",
        )
        return infers

    def init_trainers(self):
        return {
            "deepgrow_2d": TrainDeepgrow(
                model_dir=self.model_dir_2d,
                network=load_from_mmar(self.mmar_2d, self.model_dir_2d),
                publish_path=self.final_model_2d,
                description="Train 2D Deepgrow model",
                dimension=2,
                roi_size=(256, 256),
                model_size=(256, 256),
                max_train_interactions=15,
                max_val_interactions=5,
                config={
                    "train_random_slices": 10,
                    "val_random_slices": 5,
                    "max_epochs": 20,
                    "train_batch_size": 16,
                    "val_batch_size": 16,
                },
            ),
            "deepgrow_3d": TrainDeepgrow(
                model_dir=self.model_dir_3d,
                network=load_from_mmar(self.mmar_3d, self.model_dir_3d),
                description="Train 3D Deepgrow model",
                dimension=3,
                roi_size=(96, 96, 96),
                model_size=(96, 96, 96),
                max_train_interactions=15,
                max_val_interactions=20,
            ),
        }

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }
