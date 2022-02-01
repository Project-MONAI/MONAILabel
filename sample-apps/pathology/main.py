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
import shutil
from typing import Dict

from lib import MyInfer, MyTrain, resnet18_cf

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import file_ext, get_basename

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.image_size = 1024
        self.patch_size = 64
        self.grid_size = (self.image_size // self.patch_size) ** 2

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Metastasis Detection - Pathology",
            description="Active Learning solution for Pathology",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "metastasis_detection": MyInfer([self.pretrained_model, self.final_model], resnet18_cf(num_nodes=0)),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "metastasis_detection": MyTrain(
                model_dir=self.model_dir,
                network=resnet18_cf(num_nodes=self.grid_size),
                image_size=self.image_size,
                patch_size=self.patch_size,
                load_path=self.pretrained_model,
                publish_path=self.final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
                train_save_interval=1,
            )
        }


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.png"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/Camelyon/monai_dataset")
    # parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/Camelyon/dataset_v2/training/images")
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-o", "--output", default="model_01")
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "use_pretrained_model": "false",
        "auto_update_scoring": "false",
    }

    app = MyApp(app_dir, studies, conf)
    run_train = False
    if run_train:
        app.train(
            request={
                "name": args.output,
                "model": "metastasis_detection",
                "max_epochs": args.epoch,
                "dataset": "Dataset",
                "train_batch_size": 96,
                "val_batch_size": 64,
                "multi_gpu": True,
                "val_split": 0.2,
            }
        )
    else:
        req = {
            "model": "metastasis_detection",
            "image": "/local/sachi/Data/Pathology/Camelyon/monai_dataset/tumor_001_1_3x1.png",
            # "result_extension": ".jpg",
        }
        shutil.copy(req["image"], f"/local/sachi/Downloads/image{file_ext(req['image'])}")
        o = os.path.join(os.path.dirname(req["image"]), "labels", "final", get_basename(req["image"]))
        shutil.copy(o, f"/local/sachi/Downloads/original{file_ext(req['image'])}")

        res = app.infer(request=req)
        print(res)
        shutil.move(res["label"], f"/local/sachi/Downloads/label{file_ext(res['label'])}")


if __name__ == "__main__":
    main()
