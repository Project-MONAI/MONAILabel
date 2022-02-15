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
from typing import Dict

from lib import MyInfer, MyTrain
from monai.networks.nets import BasicUNet

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import file_ext, get_basename

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.patch_size = (512, 512)

        # https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv
        labels = {
            1: "tumor",
            # 2: "stroma",
            # 3: "lymphocytic_infiltrate",
            # 4: "necrosis_or_debris",
            # 5: "glandular_secretions",
            # 6: "blood",
            # 7: "exclude",
            # 8: "metaplasia_NOS",
            # 9: "fat",
            # 10: "plasma_cells",
            # 11: "other_immune_infiltrate",
            # 12: "mucoid_material",
            # 13: "normal_acinus_or_duct",
            # 14: "lymphatics",
            # 15: "undetermined",
            # 16: "nerve",
            # 17: "skin_adnexa",
            # 18: "blood_vessel",
            # 19: "angioinvasion",
            # 20: "dcis",
            # 21: "other",
        }

        self.network = BasicUNet(
            spatial_dims=2, in_channels=3, out_channels=len(labels), features=(32, 64, 128, 256, 512, 32)
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            labels=labels,
            name="Semantic Segmentation - Pathology",
            description="Active Learning solution for Pathology",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "segmentation": MyInfer([self.pretrained_model, self.final_model], self.network, labels=self.labels),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "segmentation": MyTrain(
                model_dir=self.model_dir,
                network=self.network,
                load_path=self.pretrained_model,
                publish_path=self.final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
                train_save_interval=1,
                patch_size=self.patch_size,
                labels=self.labels,
            )
        }


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.svs"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.WARN,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/BCSS/wsis")
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
                "name": "model_01",
                "model": "segmentation",
                "max_epochs": 600,
                "dataset": "PersistentDataset",
                "train_batch_size": 1,
                "val_batch_size": 1,
                "multi_gpu": True,
                "val_split": 0.1,
            }
        )
    else:
        infer_wsi(app)


def infer_wsi(app):
    root_dir = "/local/sachi/Data/Pathology/BCSS"
    image = "TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF"
    res = app.infer_wsi(
        request={
            "model": "segmentation",
            "image": image,
            "level": 0,
            "patch_size": (4096, 4096),
            "roi": ((5000, 5000), (9000, 9000)),
        }
    )

    label_json = os.path.join(root_dir, f"{image}.json")
    logger.error(f"Writing Label JSON: {label_json}")
    with open(label_json, "w") as fp:
        json.dump(res["params"], fp, indent=2)

    label_xml = os.path.join(root_dir, f"{image}.xml")
    shutil.copy(res["file"], label_xml)
    logger.error(f"Saving Label XML: {label_xml}")


def infer_roi(args, app):
    images = [os.path.join(args.studies, f) for f in os.listdir(args.studies) if f.endswith(".png")]
    # images = [os.path.join(args.studies, "tumor_001_1_4x2.png")]
    for image in images:
        print(f"Infer Image: {image}")
        req = {
            "model": "segmentation",
            "image": image,
        }

        name = get_basename(image)
        ext = file_ext(name)

        shutil.copy(image, f"/local/sachi/Downloads/image{ext}")

        o = os.path.join(os.path.dirname(image), "labels", "final", name)
        shutil.copy(o, f"/local/sachi/Downloads/original{ext}")

        res = app.infer(request=req)
        o = os.path.join(args.studies, "labels", "original", name)
        shutil.move(res["label"], o)

        shutil.copy(o, f"/local/sachi/Downloads/predicated{ext}")
        return


if __name__ == "__main__":
    main()
