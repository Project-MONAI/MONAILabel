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
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Dict

import openslide
from monai.networks.nets import BasicUNet, UNet

from lib import MyInfer, MyTrain
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
        unet = False
        if unet:
            self.network = UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=len(labels),
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        else:
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
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.png"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.WARN,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/BCSS/images")
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
        # infer_roi(args, app)
        infer_wsi(app)


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


def infer_wsi(app, image=None, level=0, patch_size=(4096, 4096), roi=((5000, 5000), (9000, 9000)), multi_thread=0):
    root_dir = "/local/sachi/Data/Pathology/BCSS"
    image = image if image else f"{root_dir}/wsis/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.svs"

    logger.error(f"Input WSI Image: {image}")
    start = time.time()

    with openslide.OpenSlide(image) as slide:
        w, h = slide.dimensions
    logger.error(f"Input WSI Image Dimensions: ({w} x {h})")

    t_cols = ceil(w / patch_size[0])  # COL
    t_rows = ceil(h / patch_size[1])  # ROW

    logger.error(f"Total Patches to infer {t_rows} x {t_cols}: {t_rows * t_cols}")
    infer_tasks = create_tasks(t_rows, t_cols, patch_size, roi)
    completed = [0] * len(infer_tasks)
    res_json = [None] * len(infer_tasks)

    infer_task = app._infers.get("segmentation")
    for device in ["cuda"]:
        if infer_task._get_network(device):
            logger.error(f"Model Loaded into {device}")
        else:
            logger.error(f"Model Not Loaded into {device}... can't run in parallel")
            return

    def run_task(task):
        tid = task["id"]
        (row, col, tx, ty, tw, th) = task["coords"]
        dev = task["device"]

        logger.info(f"{tid} => Patch/Slide ({row}, {col}) => Top: ({tx}, {ty}); Size: {tw} x {th}")
        data = {
            "model": "segmentation",
            "image": image,
            "device": dev,
            "wsi": {
                "location": (tx, ty),
                "level": level,
                "size": (tw, th)
            },
            "roi_size": patch_size,
            "result_write_to_file": False,
            "result_extension": ".png"
        }

        res = app.infer(data)
        if res.get("params") and res["params"].get("contours"):
            res_json[tid] = {"bbox": res["params"]["bbox"], "contours": res["params"]["contours"]}
        completed[tid] = 1
        logger.error(
            f"Current: {tid}; Device: {device}; Completed: {sum(completed)} / {len(completed)}; Latencies: {res['params']['latencies']}")

    logger.error(f"Total Tasks: {len(infer_tasks)}")
    if multi_thread:
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="Infer") as executor:
            executor.map(run_task, infer_tasks)
    else:
        for t in infer_tasks:
            run_task(t)

    logger.error("Infer Time Taken: {:.4f}".format(time.time() - start))

    label_json = os.path.join(root_dir, get_basename(image).replace(".svs", ".tif").replace(".tif", ".json"))
    logger.error(f"Writing Label JSON: {label_json}")
    with open(label_json, "w") as fp:
        json.dump({tid: v for tid, v in enumerate(res_json) if v}, fp, indent=2)

    convert_to_xml(label_json)
    logger.error("Total Time Taken: {:.4f}".format(time.time() - start))


def point_in_roi(p, bbox):
    return p[0] >= bbox[0][0] and p[0] <= bbox[1][0] and p[1] >= bbox[0][1] and p[1] < bbox[1][0]


def create_tasks(rows, cols, patch_size, roi=None):
    infer_tasks = []
    count = 0
    tw, th = patch_size[0], patch_size[1]
    for row in range(rows):
        for col in range(cols):
            tx = col * tw
            ty = row * th
            if roi and not point_in_roi((tx, ty), roi) and not point_in_roi((tx + tw, ty), roi) and not point_in_roi(
                    (tx, ty + th), roi) and not point_in_roi((tx + tw, ty + th), roi):
                continue

            infer_tasks.append({"id": count, "coords": (row, col, tx, ty, tw, th), "device": "cuda"})
            count += 1
    return infer_tasks


def convert_to_xml(label_json):
    with open(label_json, "r") as fp:
        data = json.load(fp)

    label_xml = label_json.replace(".json", ".xml")
    logger.error(f"Writing Label XML: {label_xml}")

    count = 0
    with open(label_xml, "w") as fp:
        fp.write('<?xml version="1.0"?>\n')
        fp.write('<ASAP_Annotations>\n')
        fp.write('  <Annotations>\n')
        for k, v in data.items():
            contours = v["contours"]
            for count, contour in enumerate(contours):
                fp.write('    <Annotation Name="Tumor" Type="Polygon" PartOfGroup="Tumor" Color="#F4FA58">\n'.format(
                    k, count))
                fp.write('      <Coordinates>\n')
                for pcount, point in enumerate(contour):
                    fp.write('        <Coordinate Order="{}" X="{}" Y="{}" />\n'.format(pcount, point[0], point[1]))
                fp.write('      </Coordinates>\n')
                count += 1
                fp.write('    </Annotation>\n')
        fp.write('  </Annotations>\n')
        fp.write('  <AnnotationGroups>\n')
        fp.write('    <Group Name="Tumor" PartOfGroup="None" Color="#00ff00">\n')
        fp.write('      <Attributes />\n')
        fp.write('    </Group>\n')
        fp.write('  </AnnotationGroups>\n')
        fp.write('</ASAP_Annotations>\n')

    logger.error(f"Total Polygons: {count}")


if __name__ == "__main__":
    main()
