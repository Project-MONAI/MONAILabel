# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes.util
import logging
import os
import platform
from ctypes import cdll
from distutils.util import strtobool
from typing import Dict

import lib.configs

from monailabel.datastore.dsa import DSADatastore
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.class_utils import get_class_names

logger = logging.getLogger(__name__)

# For windows (preload openslide dll using file_library) https://github.com/openslide/openslide-python/pull/151
if platform.system() == "Windows":
    cdll.LoadLibrary(str(ctypes.util.find_library("libopenslide-0.dll")))


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}

        models = conf.get("models", "all")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, None)

        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="MONAILabel - Pathology",
            description="DeepLearning models for pathology",
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
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers

        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t
        return trainers


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
    studies = f"{home}/Datasets/Pathology"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies

    app = MyApp(app_dir, studies, {"roi_size": "[1024,1024]", "preload": "true"})
    if run_train:
        train_nuclick(app)
    else:
        # infer_nuclick(app)
        infer_wsi(app)


def train_nuclick(app):
    model = "nuclick"
    app.train(
        request={
            "name": "train_01",
            "model": model,
            "max_epochs": 10,
            "dataset": "PersistentDataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 128,
            "val_batch_size": 64,
            "multi_gpu": True,
            "val_split": 0.2,
            "dataset_source": "none",
            "dataset_limit": 0,
            "pretrained": False,
            "n_saved": 10,
        },
    )


def train(app):
    model = "deepedit_nuclei"  # deepedit_nuclei, segmentation_nuclei
    app.train(
        request={
            "name": "train_01",
            "model": model,
            "max_epochs": 10 if model == "deepedit_nuclei" else 30,
            "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 16,
            "val_batch_size": 12,
            "multi_gpu": True,
            "val_split": 0.1,
            "dataset_source": "pannuke",
        },
    )


def infer_nuclick(app):
    import shutil

    image = "C:\\Projects\\nuclick_torch\\test\\input_image.png"
    res = app.infer(
        # request={
        #     "model": "nuclick",
        #     "image": image,
        #     "output": "asap",
        #     "foreground": [[390, 470], [1507, 190]],
        #     "location": [0, 0],
        #     "size": [0, 0],
        #     "result_extension": ".png",
        # }
        request={
            "model": "nuclick",
            "image": "JP2K-33003-1",
            "level": 0,
            "location": [2262, 4661],
            "size": [294, 219],
            "min_poly_area": 30,
            "foreground": [[2411, 4797], [2331, 4775], [2323, 4713], [2421, 4684]],
            "background": [],
            "output": "asap",
            # 'result_extension': '.png',
        }
    )

    # print(json.dumps(res, indent=2))
    shutil.move(res["label"], "C:\\Projects\\nuclick_torch\\test\\output_image.xml")
    logger.info("All Done!")


def infer_wsi(app):
    import shutil
    from pathlib import Path

    home = str(Path.home())
    root_dir = f"{home}/Datasets/"
    image = "TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7"

    output = "dsa"

    # slide = openslide.OpenSlide(f"{app.studies}/{image}.svs")
    # img = slide.read_region((7737, 20086), 0, (2048, 2048)).convert("RGB")
    # image_np = np.array(img, dtype=np.uint8)

    req = {
        "model": "deepedit_nuclei",  # deepedit_nuclei, segmentation_nuclei
        "image": image,  # image, image_np
        "output": output,
        "logging": "error",
        "level": 0,
        "location": [0, 0],
        "size": [0, 0],
        "tile_size": [1024, 1024],
        "min_poly_area": 80,
        "gpus": "all",
        "multi_gpu": True,
    }

    res = app.infer_wsi(request=req)
    if output == "asap":
        label_xml = os.path.join(root_dir, f"{image}.xml")
        shutil.copy(res["file"], label_xml)
        logger.info(f"Saving ASAP XML: {label_xml}")
    elif output == "dsa":
        label_dsa = os.path.join(root_dir, f"{image}_dsa.json")
        shutil.copy(res["file"], label_dsa)
        logger.info(f"Saving DSA JSON: {label_dsa}")
    logger.info("All Done!")


if __name__ == "__main__":
    main()
