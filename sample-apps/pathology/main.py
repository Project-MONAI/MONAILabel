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

from lib import InferDeep, MyInfer, MyTrain, TrainDeep, TrainDeepNuke
from monai.networks.nets import BasicUNet

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.patch_size = (512, 512)

        # https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv
        labels = {
            1: "tumor",
        }

        self.seg_network = BasicUNet(
            spatial_dims=2, in_channels=3, out_channels=len(labels), features=(32, 64, 128, 256, 512, 32)
        )
        self.deep_network = BasicUNet(
            spatial_dims=2, in_channels=5, out_channels=len(labels), features=(32, 64, 128, 256, 512, 32)
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.seg_pretrained_model = os.path.join(self.model_dir, "segmentation_pretrained.pt")
        self.seg_final_model = os.path.join(self.model_dir, "segmentation.pt")

        self.deep_pretrained_model = os.path.join(self.model_dir, "deepedit_pretrained.pt")
        self.deep_final_model = os.path.join(self.model_dir, "deepedit.pt")

        self.deep_nuke_pretrained_model = os.path.join(self.model_dir, "deepedit_nuke_pretrained.pt")
        self.deep_nuke_final_model = os.path.join(self.model_dir, "deepedit_nuke.pt")

        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "true"))
        seg_pretrained_model_uri = conf.get(
            "seg_pretrained_model_path", f"{self.PRE_TRAINED_PATH}pathology_segmentation_tumor.pt"
        )
        deep_pretrained_model_uri = conf.get(
            "deep_pretrained_model_path", f"{self.PRE_TRAINED_PATH}pathology_deepedit_tumor.pt"
        )
        deep_nuke_pretrained_model_uri = conf.get(
            "deep_nuke_pretrained_model_path", f"{self.PRE_TRAINED_PATH}pathology_deepedit_nuke.pt"
        )

        # Path to pretrained weights
        if use_pretrained_model:
            logger.info(f"Segmentation Pretrained Model Path: {seg_pretrained_model_uri}")
            logger.info(f"Deepedit Pretrained Model Path: {seg_pretrained_model_uri}")
            self.download(
                [
                    (self.seg_pretrained_model, seg_pretrained_model_uri),
                    (self.deep_pretrained_model, deep_pretrained_model_uri),
                    # (self.deep_nuke_final_model, deep_nuke_pretrained_model_uri),
                ]
            )

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            labels=labels,
            name="pathology",
            description="Active Learning solution for Pathology using Semantic Segmentation/Interaction (DeepEdit)",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "segmentation": MyInfer(
                [self.seg_pretrained_model, self.seg_final_model], self.seg_network, labels=self.labels
            ),
            "deepedit": InferDeep(
                [self.deep_pretrained_model, self.deep_final_model], self.deep_network, labels=self.labels
            ),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "segmentation": MyTrain(
                model_dir=os.path.join(self.model_dir, "segmentation"),
                network=self.seg_network,
                load_path=self.seg_pretrained_model,
                publish_path=self.seg_final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
                train_save_interval=1,
                patch_size=self.patch_size,
                labels=self.labels,
            ),
            "deepedit": TrainDeep(
                model_dir=os.path.join(self.model_dir, "deepedit"),
                network=self.deep_network,
                load_path=self.deep_pretrained_model,
                publish_path=self.deep_final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
                max_train_interactions=10,
                max_val_interactions=5,
                val_interval=1,
                train_save_interval=1,
                patch_size=self.patch_size,
                labels=self.labels,
            ),
            "deepedit_nuke": TrainDeepNuke(
                model_dir=os.path.join(self.model_dir, "deepedit_nuke"),
                network=self.deep_network,
                load_path=self.deep_pretrained_model,
                publish_path=self.deep_final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
                max_train_interactions=10,
                max_val_interactions=5,
                val_interval=1,
                train_save_interval=1,
                patch_size=(256, 256),
                labels={
                    0: "Neoplastic cells",
                    1: "Inflammatory",
                    2: "Connective/Soft tissue cells",
                    3: "Dead Cells",
                    4: "Epithelial",
                },
            ),
        }


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.svs", "*.png", "*.npy"]
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
                "model": "deepedit_nuke",
                "max_epochs": 300,
                "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
                "train_batch_size": 16,
                "val_batch_size": 12,
                "multi_gpu": True,
                "val_split": 0.1,
            }
        )
    else:
        infer_wsi(app)


def infer_wsi(app):
    root_dir = "/local/sachi/Data/Pathology/BCSS"
    image = "TCGA-EW-A1OW-01Z-00-DX1.97888686-EBB6-4B13-AB5D-452F475E865B"
    res = app.infer_wsi(
        request={
            "model": "deepedit",
            "image": image,
            "level": 0,
            "patch_size": [4096, 4096],
            "roi": {"x": 34679, "y": 54441, "x2": 46025, "y2": 64181},
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
