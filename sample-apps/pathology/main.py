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

from monailabel.datastore.dsa import DSADatastore
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.label_colors = {
            "Neoplastic cells": (255, 0, 0),
            "Inflammatory": (255, 255, 0),
            "Connective/Soft tissue cells": (0, 255, 0),
            "Dead Cells": (0, 0, 0),
            "Epithelial": (0, 0, 255),
            "Nuclei": (0, 255, 255),
        }

        self.deep_labels = ["Nuclei"]
        self.seg_labels = {
            "Neoplastic cells": 1,
            "Inflammatory": 2,
            "Connective/Soft tissue cells": 3,
            "Dead Cells": 4,
            "Epithelial": 5,
        }

        self.seg_network = BasicUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=len(self.seg_labels) + 1 if len(self.seg_labels) > 1 else 1,
            features=(32, 64, 128, 256, 512, 32),
        )
        self.deepedit_network = BasicUNet(
            spatial_dims=2,
            in_channels=5,
            out_channels=len(self.deep_labels) + 1 if len(self.deep_labels) > 1 else 1,
            features=(32, 64, 128, 256, 512, 32),
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.seg_pretrained_model = os.path.join(self.model_dir, "segmentation_nuclei_pretrained.pt")
        self.seg_final_model = os.path.join(self.model_dir, "segmentation_nuclei.pt")

        self.deepedit_pretrained_model = os.path.join(self.model_dir, "deepedit_nuclei_pretrained.pt")
        self.deepedit_final_model = os.path.join(self.model_dir, "deepedit_nuclei.pt")

        use_pretrained_model = strtobool(conf.get("use_pretrained_model", "true"))
        seg_pretrained_model_uri = f"{self.PRE_TRAINED_PATH}/pathology_segmentation_nuclei.pt"
        deepedit_pretrained_model_uri = f"{self.PRE_TRAINED_PATH}/pathology_deepedit_nuclei.pt"

        # Path to pretrained weights
        if use_pretrained_model:
            logger.info(f"++ Segmentation Pretrained Model Path: {seg_pretrained_model_uri}")
            logger.info(f"++ DeepEdit Pretrained Model Path: {deepedit_pretrained_model_uri}")
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
            name="pathology",
            description="Active Learning solution for Nuclei Instance Segmentation",
        )

    def init_remote_datastore(self) -> Datastore:
        """
        -s http://0.0.0.0:8080/api/v1
        -c dsa_folder 621e94e2b6881a7a4bef5170
        -c dsa_api_key OJDE9hjuOIS6R8oEqhnVYHUpRpk18NfJABMt36dJ
        -c dsa_asset_store_path /localhome/sachi/Projects/digital_slide_archive/devops/dsa/assetstore
        """

        logger.info(f"Using DSA Server: {self.studies}")
        folder = self.conf.get("dsa_folder")
        annotation_groups = self.conf.get("dsa_groups", None)
        api_key = self.conf.get("dsa_api_key")
        asset_store_path = self.conf.get("dsa_asset_store_path")

        return DSADatastore(
            api_url=self.studies,
            folder=folder,
            api_key=api_key,
            annotation_groups=annotation_groups,
            asset_store_path=asset_store_path,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "segmentation": InferSegmentation(
                path=[self.seg_pretrained_model, self.seg_final_model],
                network=self.seg_network,
                labels=self.seg_labels,
                label_colors=self.label_colors,
            ),
            "deepedit": InferDeepedit(
                path=[self.deepedit_pretrained_model, self.deepedit_final_model],
                network=self.deepedit_network,
                labels=self.deep_labels,
                label_colors=self.label_colors,
            ),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        config = {
            "max_epochs": 10,
            "train_batch_size": 1,
            "dataset_max_region": (10240, 10240),
            "dataset_limit": 0,
            "dataset_randomize": True,
        }
        return {
            "segmentation": TrainSegmentation(
                model_dir=os.path.join(self.model_dir, "segmentation"),
                network=self.seg_network,
                load_path=self.seg_pretrained_model,
                publish_path=self.seg_final_model,
                config=config,
                train_save_interval=1,
                labels=self.seg_labels,
            ),
            "deepedit": TrainDeepEdit(
                model_dir=os.path.join(self.model_dir, "deepedit"),
                network=self.deepedit_network,
                load_path=self.deepedit_pretrained_model,
                publish_path=self.deepedit_final_model,
                config=config,
                max_train_interactions=10,
                max_val_interactions=5,
                val_interval=1,
                train_save_interval=1,
                labels=self.deep_labels,
            ),
        }


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse
    from pathlib import Path

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.svs", "*.png", "*.npy", "*.tif", ".xml"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_train = False
    home = str(Path.home())
    if run_train:
        # studies = f"{home}/Data/Pathology/PanNuke"
        studies = "http://0.0.0.0:8080/api/v1"
    else:
        studies = f"{home}/Data/Pathology/Test"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies

    app = MyApp(app_dir, studies, {})
    model = "deepedit"  # deepedit, segmentation
    if run_train:
        app.train(
            request={
                "name": "model_01",
                "model": model,
                "max_epochs": 10 if model == "deepedit" else 30,
                "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
                "train_batch_size": 16,
                "val_batch_size": 12,
                "multi_gpu": True,
                "val_split": 0.1,
                "dataset_source": "pannuke",
            },
        )
    else:
        infer_wsi(app)


def infer_wsi(app):
    from pathlib import Path

    import numpy as np
    import openslide

    home = str(Path.home())

    root_dir = f"{home}/Data/Pathology"
    image = "TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7"

    output = "dsa"

    slide = openslide.OpenSlide(f"{app.studies}/{image}.svs")
    img = slide.read_region((7737, 20086), 0, (2048, 2048)).convert("RGB")
    image_np = np.array(img, dtype=np.uint8)

    res = app.infer_wsi(
        request={
            "model": "deepedit",  # deepedit, segmentation
            "image": image,  # image, image_np
            "output": output,
            "logging": "error",
            "level": 0,
            "location": [7737, 20086],
            "size": [5522, 3311],
            "tile_size": [2048, 2048],
            "min_poly_area": 40,
            "gpus": "all",
        }
    )

    label_json = os.path.join(root_dir, f"{image}.json")
    logger.info(f"Writing Label JSON: {label_json}")
    with open(label_json, "w") as fp:
        json.dump(res["params"], fp, indent=2)

    if output == "asap":
        label_xml = os.path.join(root_dir, f"{image}.xml")
        shutil.copy(res["file"], label_xml)
        logger.info(f"Saving ASAP XML: {label_xml}")
    elif output == "dsa":
        label_dsa = os.path.join(root_dir, f"{image}_dsa.json")
        shutil.copy(res["file"], label_dsa)
        logger.info(f"Saving DSA JSON: {label_dsa}")


if __name__ == "__main__":
    main()
