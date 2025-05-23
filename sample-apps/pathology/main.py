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
from typing import Dict

import lib.configs
from lib.activelearning.random import WSIRandom
from lib.infers import NuClick
from lib.transforms import LoadImagePatchd

import monailabel
from monailabel.datastore.dsa import DSADatastore
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.sam2.utils import is_sam2_module_available
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import strtobool

logger = logging.getLogger(__name__)

# For windows (preload openslide dll using file_library) https://github.com/openslide/openslide-python/pull/151
if platform.system() == "Windows":
    cdll.LoadLibrary(str(ctypes.util.find_library("libopenslide-0.dll")))


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        configs = {}
        candidates = get_class_names(lib.configs, "TaskConfig")
        for c in candidates:
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}

        models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            # exit(-1)

        models = models.split(",") if models else []
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

        self.sam = strtobool(conf.get("sam2", "true"))
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Pathology ({monailabel.__version__})",
            description="DeepLearning models for pathology",
            version=monailabel.__version__,
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
            api_key=api_key,
            folder=folder,
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

        #################################################
        # Pipeline based on existing infers
        #################################################
        if infers.get("nuclick") and infers.get("classification_nuclei"):
            p = infers["nuclick"]
            c = infers["classification_nuclei"]
            if isinstance(p, NuClick) and isinstance(c, BasicInferTask):
                p.init_classification(c)

        #################################################
        # SAM
        #################################################
        if is_sam2_module_available() and self.sam:
            from monailabel.sam2.infer import Sam2InferTask

            infers["sam_2d"] = Sam2InferTask(
                model_dir=self.model_dir,
                type=InferType.ANNOTATION,
                dimension=2,
                additional_info={"nuclick": True, "pathology": True},
                image_loader=LoadImagePatchd(keys="image", padding=False),
                post_trans=[FindContoursd(keys="pred")],
                writer=PolygonWriter(),
                config={"cache_image": False, "reset_state": True},
            )

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

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "wsi_random": WSIRandom(),
        }

        if strtobool(self.conf.get("skip_strategies", "false")):
            return strategies

        for n, task_config in self.models.items():
            s = task_config.strategy()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Strategy:: {k} => {v}")
                strategies[k] = v

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    from pathlib import Path

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_READ_ONLY = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.svs", "*.png", "*.npy", "*.tif", ".xml"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Dataset/Pathology/dummy"

    app_dir = os.path.dirname(__file__)
    app = MyApp(
        app_dir,
        studies,
        {
            "preload": "true",
            "models": "hovernet_nuclei",
            "use_pretrained_model": "false",
            "consep": "true",
        },
    )

    # train_from_dataset(app, "nuclick", "Nuclei")
    # infer(app, "hovernet_nuclei")
    train(app, "hovernet_nuclei")


def train_from_dataset(app, model, postfix):
    import json
    import random
    from pathlib import Path

    from monailabel.utils.others.generic import create_dataset_from_path

    home = str(Path.home())
    train_dir = f"{home}/Dataset/Pathology/CoNSeP/training{postfix}"
    val_dir = f"{home}/Dataset/Pathology/CoNSeP/validation{postfix}"

    train_ds = create_dataset_from_path(train_dir, img_ext=".png", image_dir="", label_dir="labels/final")
    val_ds = create_dataset_from_path(val_dir, img_ext=".png", image_dir="", label_dir="labels/final")
    random.shuffle(train_ds)
    random.shuffle(val_ds)

    # train_ds = train_ds[:1024]
    # val_ds = val_ds[:64]

    train_ds_json = f"{home}/Dataset/Pathology/CoNSeP/train_ds.json"
    val_ds_json = f"{home}/Dataset/Pathology/CoNSeP/val_ds.json"

    with open(train_ds_json, "w") as fp:
        json.dump(train_ds, fp, indent=2)
    with open(val_ds_json, "w") as fp:
        json.dump(val_ds, fp, indent=2)

    app.train(
        request={
            "name": "train_01",
            "model": model,
            "max_epochs": 50,
            "dataset": "PersistentDataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 128,
            "val_batch_size": 128,
            "multi_gpu": False,
            "val_split": 0.2,
            "dataset_source": "none",
            "dataset_limit": 0,
            "pretrained": False,
            "n_saved": 10,
            "train_ds": train_ds_json,
            "val_ds": val_ds_json,
        },
    )


def train(app, model):
    app.train(
        request={
            "name": "train_01",
            "model": model,
            "max_epochs": 10,
            "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 16,
            "val_batch_size": 16,
            "multi_gpu": True,
            "val_split": 0.2,
            "dataset_limit": 0,
            "pretrained": True,
        },
    )


def infer(app, model):
    # import json

    request = {
        "model": model,
        "image": "test_1",
        "output": "json",
    }
    res = app.infer(request)
    # print(json.dumps(res, indent=2))


def infer_nuclick(app, classify=True):
    import shutil

    request = {
        "model": "nuclick",
        "image": "JP2K-33003-1",
        "level": 0,
        "location": [2262, 4661],
        "size": [294, 219],
        "min_poly_area": 30,
        "foreground": [[2411, 4797], [2331, 4775], [2323, 4713], [2421, 4684]],
        "background": [],
        "output": "json" if classify else "asap",
    }

    res = app.infer(request)
    shutil.move(res["label"], os.path.join(app.studies, "..", "output_image.xml"))
    logger.info("All Done!")


def infer_nuclick_classification(app):
    import shutil

    request = {
        "model": "nuclick_classification",
        "image": "JP2K-33003-1",
        "output": "asap",
        "level": 0,
        "location": [2387, 4845],
        "size": [215, 165],
        "tile_size": [1024, 1024],
        "min_poly_area": 30,
        "foreground": [[2486, 4870], [2534, 4941], [2500, 4947], [2418, 4936], [2462, 4979], [2429, 4976]],
        "background": [],
    }

    res = app.infer(request)
    shutil.move(res["label"], os.path.join(app.studies, "..", "output_image.xml"))
    logger.info("All Done!")


def infer_wsi(app):
    import shutil

    image = "TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7"
    output = "dsa"

    # slide = openslide.OpenSlide(f"{app.studies}/{image}.svs")
    # img = slide.read_region((7737, 20086), 0, (2048, 2048)).convert("RGB")
    # image_np = np.array(img, dtype=np.uint8)

    req = {
        "model": "segmentation_nuclei",
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

    root_dir = os.path.join(app.studies, "..")

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
