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
import shutil
from distutils.util import strtobool
from typing import Dict

from lib import InferDeepedit, InferSegmentation, TrainDeepEdit, TrainSegmentation
from monai.networks.nets import BasicUNet

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        labels = {
            0: "Neoplastic cells",
            1: "Inflammatory",
            2: "Connective/Soft tissue cells",
            3: "Dead Cells",
            4: "Epithelial",
        }

        self.seg_network = BasicUNet(
            spatial_dims=2, in_channels=3, out_channels=len(labels) + 1, features=(32, 64, 128, 256, 512, 32)
        )
        self.deepedit_network = BasicUNet(
            spatial_dims=2, in_channels=5, out_channels=1, features=(32, 64, 128, 256, 512, 32)
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.seg_pretrained_model = os.path.join(self.model_dir, "segmentation_nuclei_pretrained.pt")
        self.seg_final_model = os.path.join(self.model_dir, "segmentation.pt")

        self.deepedit_pretrained_model = os.path.join(self.model_dir, "deepedit_nuclei_pretrained.pt")
        self.deepedit_final_model = os.path.join(self.model_dir, "deepedit.pt")

        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "false"))
        seg_pretrained_model_uri = conf.get(
            "seg_pretrained_model_path", f"{self.PRE_TRAINED_PATH}pathology_segmentation_nuclei.pt"
        )
        deepedit_pretrained_model_uri = conf.get(
            "deepedit_pretrained_model_path", f"{self.PRE_TRAINED_PATH}pathology_deepedit_nuclei.pt"
        )

        # Path to pretrained weights
        if use_pretrained_model:
            logger.info(f"Segmentation Pretrained Model Path: {seg_pretrained_model_uri}")
            logger.info(f"Deepedit Pretrained Model Path: {seg_pretrained_model_uri}")
            self.download(
                [
                    (self.seg_pretrained_model, seg_pretrained_model_uri),
                    (self.deepedit_pretrained_model, deepedit_pretrained_model_uri),
                ]
            )

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            labels=labels,
            name="pathology",
            description="Active Learning solution for Nuclei Instance Segmentation",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "segmentation": InferSegmentation(
                [self.seg_pretrained_model, self.seg_final_model], self.seg_network, labels=self.labels
            ),
            "deepedit": InferDeepedit(
                [self.deepedit_pretrained_model, self.deepedit_final_model], self.deepedit_network, labels=self.labels
            ),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "segmentation": TrainSegmentation(
                model_dir=os.path.join(self.model_dir, "segmentation"),
                network=self.seg_network,
                load_path=self.seg_pretrained_model,
                publish_path=self.seg_final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
                train_save_interval=1,
                labels=self.labels,
            ),
            "deepedit": TrainDeepEdit(
                model_dir=os.path.join(self.model_dir, "deepedit"),
                network=self.deepedit_network,
                load_path=self.deepedit_pretrained_model,
                publish_path=self.deepedit_final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
                max_train_interactions=10,
                max_val_interactions=5,
                val_interval=1,
                train_save_interval=1,
                labels=self.labels,
            ),
        }


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.svs", "*.png", "*.npy", "*.tif"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/PanNukeF")
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "use_pretrained_model": "false",
        "auto_update_scoring": "false",
    }

    app = MyApp(app_dir, studies, conf)
    run_train = True
    if run_train:
        app.train(
            request={
                "name": "model_01",
                "model": "segmentation",
                "max_epochs": 500,
                "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
                "train_batch_size": 16,
                "val_batch_size": 12,
                "multi_gpu": True,
                "val_split": 0.1,
            },
            # request={
            #     "name": "model_01",
            #     "model": "deepedit",
            #     "max_epochs": 50,
            #     "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
            #     "train_batch_size": 16,
            #     "val_batch_size": 12,
            #     "multi_gpu": True,
            #     "val_split": 0.1,
            # },
        )
    else:
        infer_wsi(app)


def infer_wsi(app):
    root_dir = "/local/sachi/Data/Pathology/BCSS/wsis"
    image = "TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7"
    res = app.infer_wsi(
        request={
            "model": "deepedit",
            "image": image,
            "level": 0,
            "patch_size": [2048, 2048],
            "roi": {"x": 7737, "y": 20086, "x2": 9785, "y2": 22134},
            "min_poly_area": 40,
        }
    )

    label_json = os.path.join(root_dir, f"{image}.json")
    logger.error(f"Writing Label JSON: {label_json}")
    with open(label_json, "w") as fp:
        json.dump(res["params"], fp, indent=2)

    label_xml = os.path.join(root_dir, f"{image}.xml")
    shutil.copy(res["file"], label_xml)
    logger.error(f"Saving Label XML: {label_xml}")


if __name__ == "__main__":
    main()
