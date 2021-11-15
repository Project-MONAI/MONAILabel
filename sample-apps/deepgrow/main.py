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
from typing import Dict

from lib import InferDeepgrow, MyStrategy, TrainDeepgrow
from monai.apps import load_from_mmar

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.scribbles.infer import HistogramBasedGraphCut
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.infer.deepgrow_pipeline import InferDeepgrowPipeline

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
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
            conf=conf,
            name="Deepgrow - Generic",
            description="Active Learning solution to label generic organ",
            labels=[
                "spleen",
                "right kidney",
                "left kidney",
                "gallbladder",
                "esophagus",
                "liver",
                "stomach",
                "aorta",
                "inferior vena cava",
                "portal vein and splenic vein",
                "pancreas",
                "right adrenal gland",
                "left adrenal gland",
            ],
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers = {
            "deepgrow_2d": InferDeepgrow(self.final_model_2d, load_from_mmar(self.mmar_2d, self.model_dir_2d)),
            "deepgrow_3d": InferDeepgrow(
                self.final_model_3d,
                load_from_mmar(self.mmar_3d, self.model_dir_3d),
                dimension=3,
                model_size=(128, 192, 192),
            ),
            # intensity range set for CT Soft Tissue
            "Histogram+GraphCut": HistogramBasedGraphCut(
                intensity_range=(-300, 200, 0.0, 1.0, True), pix_dim=(2.5, 2.5, 5.0), lamda=1.0, sigma=0.1
            ),
        }

        infers["deepgrow_pipeline"] = InferDeepgrowPipeline(
            path=None,
            network=load_from_mmar(self.mmar_2d, self.model_dir_2d),
            model_3d=infers["deepgrow_3d"],
            description="Combines Deepgrow 2D model and 3D deepgrow model",
        )
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "deepgrow_2d": TrainDeepgrow(
                model_dir=self.model_dir_2d,
                network=load_from_mmar(self.mmar_2d, self.model_dir_2d),
                publish_path=self.final_model_2d,
                description="Train 2D Deepgrow model",
                dimension=2,
                roi_size=(256, 256),
                model_size=(256, 256),
                max_train_interactions=10,
                max_val_interactions=5,
                val_interval=5,
                config={
                    "max_epochs": 10,
                    "train_batch_size": 16,
                    "val_batch_size": 16,
                    "to_gpu": True,
                },
            ),
            "deepgrow_3d": TrainDeepgrow(
                model_dir=self.model_dir_3d,
                network=load_from_mmar(self.mmar_3d, self.model_dir_3d),
                description="Train 3D Deepgrow model",
                dimension=3,
                roi_size=(128, 192, 192),
                model_size=(128, 192, 192),
                max_train_interactions=15,
                max_val_interactions=10,
                val_interval=5,
                config={
                    "to_gpu": True,
                },
            ),
        }

    def init_strategies(self) -> Dict[str, Strategy]:
        return {
            "random": Random(),
            "first": MyStrategy(),
        }


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
    parser.add_argument("-s", "--studies", default="/local/sachi/Datasets/Test")
    parser.add_argument("-e", "--epoch", type=int, default=3)
    parser.add_argument("-d", "--dataset", default="CacheDataset")
    parser.add_argument("-o", "--output", default="model_01")
    parser.add_argument("-b", "--batch", type=int, default=1)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "use_pretrained_model": "true",
        "auto_update_scoring": "false",
    }

    app = MyApp(app_dir, studies, conf)
    app.train(
        request={
            "name": args.output,
            "model": "deepgrow_3d",
            "max_epochs": args.epoch,
            "dataset": args.dataset,
            "train_batch_size": args.batch,
            "multi_gpu": True,
        }
    )


if __name__ == "__main__":
    main()
