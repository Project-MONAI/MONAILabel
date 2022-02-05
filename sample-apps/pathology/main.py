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
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Dict

import cv2
import numpy as np
import openslide
import pyvips
from lib import MyInfer, MyTrain
from monai.networks.nets import UNet
from PIL import Image

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import file_ext, get_basename

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.patch_size = (512, 512)

        # https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv
        self.labels = {
            1: "tumor",
            2: "stroma",
            3: "lymphocytic_infiltrate",
            4: "necrosis_or_debris",
            5: "glandular_secretions",
            6: "blood",
            7: "exclude",
            8: "metaplasia_NOS",
            9: "fat",
            10: "plasma_cells",
            11: "other_immune_infiltrate",
            12: "mucoid_material",
            13: "normal_acinus_or_duct",
            14: "lymphatics",
            15: "undetermined",
            16: "nerve",
            17: "skin_adnexa",
            18: "blood_vessel",
            19: "angioinvasion",
            20: "dcis",
            21: "other",
        }
        self.network = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

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
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/BCSS/images")
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
    run_train = True
    if run_train:
        app.train(
            request={
                "name": "model_01",
                "model": "segmentation",
                "max_epochs": 500,
                "dataset": "Dataset",
                "train_batch_size": 2,
                "val_batch_size": 2,
                "multi_gpu": True,
                "val_split": 0.2,
            }
        )
    else:
        # infer_roi(args, app)
        # merge_labels(args)
        infer_wsi(app)


def infer_roi(args, app):
    images = [os.path.join(args.studies, f) for f in os.listdir(args.studies) if f.endswith(".png")]
    # images = [os.path.join(args.studies, "tumor_001_1_4x2.png")]
    for image in images:
        print(f"Infer Image: {image}")
        req = {
            "model": "metastasis_detection",
            "image": image,
        }

        name = get_basename(image)
        ext = file_ext(name)

        # shutil.copy(image, f"/local/sachi/Downloads/image{ext}")

        # o = os.path.join(os.path.dirname(image), "labels", "final", name)
        # shutil.copy(o, f"/local/sachi/Downloads/original{ext}")

        res = app.infer(request=req)
        o = os.path.join(args.studies, "labels", "original", name)
        shutil.move(res["label"], o)

        # shutil.copy(o, f"/local/sachi/Downloads/predicated{ext}")
        # return


def merge_labels(args):
    labels_dir = os.path.join(args.studies, "labels", "original")
    labels = sorted([f for f in os.listdir(labels_dir) if f.endswith(".png")])

    d = {}
    for label in labels:
        l = label.replace(file_ext(label), "").split("_")  # "tumor_001_0_0x0.png"
        name = f"{l[0]}_{l[1]}"
        idx = int(l[2])
        tj = int(l[3].split("x")[0])
        ti = int(l[3].split("x")[1])

        if d.get(name) is None:
            d[name] = {}
        if d[name].get(idx) is None:
            d[name][idx] = {}
        if d[name][idx].get(tj) is None:
            d[name][idx][tj] = []
        d[name][idx][tj].append(label)

        # print(f"Name: {name} => {idx} => {tj}x{ti}")
    print(d)
    for name in d:
        for idx in d[name]:
            r = len(d[name][idx].keys())
            print(d[name][idx])
            c = len(d[name][idx][0])
            print(f"grid for {idx} = {r} x {c}")

            label_np = np.zeros((r * 1024, c * 1024), dtype=np.uint8)
            for i in d[name][idx]:
                for j in range(len(d[name][idx][i])):
                    img = Image.open(os.path.join(labels_dir, d[name][idx][i][j]))
                    sx = i * 1024
                    sy = j * 1024
                    label_np[sx : (sx + 1024), sy : (sy + 1024)] = np.array(img)
            cv2.imwrite(os.path.join(args.studies, "labels", f"o_{name}_{idx}.jpg"), label_np)

    # os.path.join(args.studies, "labels", "original")


def infer_wsi(app):
    root_dir = "/local/sachi/Data/Pathology/Camelyon"
    image = f"{root_dir}/79397/training/images/tumor/tumor_001.tif"

    patch_size = [1024, 1024]
    task = app._infers.get("metastasis_detection")
    devices = ["cuda"]  # Not able to use multi-gpu for inference
    for device in devices:
        if task._get_network(device):
            logger.error(f"Model Loaded into {device}")
        else:
            logger.error(f"Model Not Loaded into {device}... can't run in parallel")
            return

    slide = openslide.OpenSlide(image)
    logger.info(f"Slide : {slide.dimensions}")
    start = time.time()

    level = 0
    w, h = slide.dimensions
    max_w = patch_size[0]
    max_h = patch_size[1]

    tiles_i = ceil(w / max_w)  # COL
    tiles_j = ceil(h / max_h)  # ROW

    logger.error(f"Total Patches to infer {tiles_i} x {tiles_j}: {tiles_i * tiles_j}")
    label_np = np.zeros((w, h), dtype=np.uint8)

    infer_tasks, completed = create_tasks(tiles_j, tiles_i, w, h, max_w, max_h)

    def run_task(t):
        batches = []
        batch_coords = []
        tid = t["id"]
        dev = devices[tid % len(devices)]
        for c in t["coords"]:
            (tj, ti, tx, ty, tw, th) = c
            logger.info(f"Patch/Slide ({tj}, {ti}) => Top: ({tx}, {ty}); Size: {tw} x {th}")
            region_rgb = slide.read_region((tx, ty), level, (tw, th)).convert("RGB")
            if region_rgb.size[0] != patch_size[0] or region_rgb.size[1] != patch_size[1]:
                logger.info("Ignore this region... (Add padding later)")
                continue
            batches.append(region_rgb)
            batch_coords.append((tx, ty, tw, th))

        _, res = task({"image": batches, "result_write_to_file": False, "device": dev})
        for bidx in range(len(batches)):
            tx, ty, tw, th = batch_coords[bidx]
            label_np[tx : (tx + tw), ty : (ty + th)] = res["pred"][bidx]

        completed[tid] = 1
        logger.error(f"Current: {tid}; Device: {dev}; Completed: {sum(completed)} / {len(completed)}")

    logger.error(f"Total Tasks: {len(infer_tasks)}")
    multi_thread = False
    if multi_thread:
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="Infer") as executor:
            executor.map(run_task, infer_tasks)
    else:
        for t in infer_tasks:
            run_task(t)

    logger.error("Infer Time Taken: {:.4f}".format(time.time() - start))
    label_file = os.path.join(root_dir, "label_batched_thread.tif")

    logger.error(f"Saving Label PNG")
    img = Image.fromarray(label_np).convert("RGB")
    img.save(os.path.join(root_dir, "label_batched_thread.png"))

    logger.error(f"Creating Label: {label_file}")
    linear = label_np.reshape(-1)
    im = pyvips.Image.new_from_memory(linear.data, label_np.shape[1], label_np.shape[0], bands=1, format="uchar")

    logger.error(f"Writing Label: {label_file}")
    im.write_to_file(label_file, pyramid=True, bigtiff=True, tile=True, tile_width=512, tile_height=512)
    logger.error("Total Time Taken: {:.4f}".format(time.time() - start))


def create_tasks(tiles_j, tiles_i, w, h, max_w, max_h):
    coords = []
    infer_tasks = []
    batch_size = 32
    completed = []

    for tj in range(tiles_j):
        for ti in range(tiles_i):
            tw = min(max_w, w - ti * max_w)
            th = min(max_h, h - tj * max_h)

            tx = ti * max_w
            ty = tj * max_h

            coords.append((tj, ti, tx, ty, tw, th))
            if len(coords) == batch_size:
                infer_tasks.append({"id": len(completed), "coords": coords})
                completed.append(0)
                coords = []
                # return infer_tasks, completed

    # Run Last Batch
    if len(coords):
        infer_tasks.append({"id": len(completed), "coords": coords})
        completed.append(0)
    return infer_tasks, completed


if __name__ == "__main__":
    main()
