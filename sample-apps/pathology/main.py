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

import numpy as np
import openslide
import pyvips
import torch
from lib import MyInfer, MyTrain
from monai.networks.nets import BasicUNet, UNet
from monai.transforms import ScaleIntensity, rescale_array
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
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/BCSS/monai")
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
        infer_roi(args, app)
        # infer_wsi(app)
        # infer_wsi_small(app)


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


def infer_wsi_small(app):
    root_dir = "/local/sachi/Data/Pathology/BCSS"
    # image = f"{root_dir}/wsis/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.svs"
    image = f"{root_dir}/wsis/TCGA-AC-A6IW-01Z-00-DX1.C4514189-E64F-4603-8970-230FA2BB77FC.svs"

    task = app._infers.get("segmentation")
    devices = ["cuda"]
    for device in devices:
        if task._get_network(device):
            logger.error(f"Model Loaded into {device}")
        else:
            logger.error(f"Model Not Loaded into {device}... can't run in parallel")
            return

    logger.info(f"Input WSIS Image: {image}")
    slide = openslide.OpenSlide(image)
    logger.info(f"Slide : {slide.dimensions}")
    start = time.time()

    level = 0
    device = devices[0]
    w, h = slide.dimensions
    region_rgb = slide.read_region((0, 0), level, (w, h)).convert("RGB")

    # region_rgb.save(os.path.join(res_dir, f"{tid}_{len(batches)}_img.png"))
    scaler = ScaleIntensity()
    image_np = scaler(np.array(region_rgb, np.uint8).transpose((2, 0, 1)))
    logger.info(f"Input Image: {image_np.shape}")

    res = task.run_inferer(data={"image": image_np}, device=device)
    p = torch.sigmoid(res["pred"][0]).detach().cpu().numpy()
    p[p > 0.5] = 255
    p = p[0] if len(p.shape) >= 3 else p
    p = p.astype(dtype=np.uint8)

    logger.info(f"Output Pred: {p.shape}; {p.dtype}")

    logger.info("Infer Time Taken: {:.4f}".format(time.time() - start))
    label_file = os.path.join(root_dir, "label.tif")

    logger.info(f"Saving Label PNG")
    img = Image.fromarray(p).convert("RGB")
    img.save(os.path.join(root_dir, "label.png"))

    logger.info(f"Creating Label: {label_file}")
    logger.info(f"Writing Label: {label_file}; shape: {p.shape}")

    linear = p.reshape(-1)
    im = pyvips.Image.new_from_memory(linear.data, p.shape[1], p.shape[0], bands=1, format="uchar")
    im.write_to_file(label_file, pyramid=True, bigtiff=True, tile=True, compression="jpeg")

    logger.info(f"TIF-Label dimensions: {openslide.OpenSlide(label_file).dimensions}")
    logger.info("Total Time Taken: {:.4f}".format(time.time() - start))


def infer_wsi(app):
    root_dir = "/local/sachi/Data/Pathology/BCSS"
    image = f"{root_dir}/wsis/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.svs"
    # image = f"{root_dir}/wsis/TCGA-A7-A6VY-01Z-00-DX1.38D4EBD7-40B0-4EE3-960A-1F00E8F83ADB.svs"
    # image = f"{root_dir}/wsis/TCGA-AC-A6IW-01Z-00-DX1.C4514189-E64F-4603-8970-230FA2BB77FC.svs"

    batch_size = 8
    patch_size = (1024, 1024)
    task = app._infers.get("segmentation")
    task.sliding_window = False
    devices = ["cuda"]  # Not able to use multi-gpu for inference
    for device in devices:
        if task._get_network(device):
            logger.error(f"Model Loaded into {device}")
        else:
            logger.error(f"Model Not Loaded into {device}... can't run in parallel")
            return

    logger.error(f"Input WSIS Image: {image}")
    slide = openslide.OpenSlide(image)
    logger.error(f"Slide : {slide.dimensions}")
    start = time.time()

    level = 0
    w, h = slide.dimensions
    max_w = patch_size[0]
    max_h = patch_size[1]

    tiles_i = ceil(w / max_w)  # COL
    tiles_j = ceil(h / max_h)  # ROW

    logger.error(f"Total Patches to infer {tiles_i} x {tiles_j}: {tiles_i * tiles_j}")
    label_np = np.zeros((h, w), dtype=np.uint8)

    infer_tasks, completed = create_tasks(batch_size, tiles_j, tiles_i, w, h, max_w, max_h)
    res_dir = os.path.join(root_dir, "result")
    os.makedirs(res_dir, exist_ok=True)

    def run_task(t):
        batches = []
        batch_coords = []
        tid = t["id"]
        device = devices[tid % len(devices)]
        scaler = ScaleIntensity()
        padded = []

        for c in t["coords"]:
            (tj, ti, tx, ty, tw, th) = c
            logger.debug(f"Patch/Slide ({tj}, {ti}) => Top: ({tx}, {ty}); Size: {tw} x {th}")

            region_rgb = slide.read_region((tx, ty), level, (tw, th)).convert("RGB")
            image_np = np.array(region_rgb, np.uint8)
            if image_np.shape[0] != patch_size[0] or image_np.shape[1] != patch_size[1]:
                background = np.zeros((patch_size[0], patch_size[1], 3), dtype=image_np.dtype)
                background[0 : image_np.shape[0], 0 : image_np.shape[1]] = image_np
                padded.append(image_np.shape[:2])
                image_np = background
            else:
                padded.append(None)

            image_np = scaler(image_np.transpose((2, 0, 1)))
            batches.append(image_np)
            batch_coords.append((tx, ty, tw, th))

        if len(batches):
            image_b = np.array(batches)
            logger.debug(f"Image Batch: {image_b.shape}")
            res = task.run_inferer(data={"image": image_b}, convert_to_batch=False, device=device)
            for bidx in range(len(batches)):
                tx, ty, tw, th = batch_coords[bidx]
                p = torch.sigmoid(res["pred"][bidx]).detach().cpu().numpy()
                p[p > 0.5] = 255
                p = p[0] if len(p.shape) >= 3 else p
                p = p.astype(dtype=np.uint8)

                logger.debug(f"Label Pred: {p.shape}")
                if padded[bidx]:
                    ox, oy = padded[bidx]
                    p = p[0:ox, 0:oy]
                label_np[ty : (ty + th), tx : (tx + tw)] = p

                image_i = Image.fromarray(rescale_array(image_b[bidx], 0, 1).transpose(1, 2, 0), "RGB")
                image_i.save(os.path.join(res_dir, f"{tid}_{bidx}_img.png"))

                label_i = Image.fromarray(p).convert("RGB")
                label_i.save(os.path.join(res_dir, f"{tid}_{bidx}_lab.png"))

        completed[tid] = 1
        logger.error(f"Current: {tid}; Device: {device}; Completed: {sum(completed)} / {len(completed)}")

    logger.error(f"Total Tasks: {len(infer_tasks)}")
    multi_thread = False
    if multi_thread:
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="Infer") as executor:
            executor.map(run_task, infer_tasks)
    else:
        for t in infer_tasks:
            run_task(t)

    logger.error("Infer Time Taken: {:.4f}".format(time.time() - start))
    label_file = os.path.join(root_dir, "label.tif")

    logger.error("Saving Label PNG")
    img = Image.fromarray(label_np).convert("RGB")
    img.save(os.path.join(root_dir, "label.png"))

    logger.error(f"Creating Label: {label_file}")
    linear = label_np.reshape(-1)
    im = pyvips.Image.new_from_memory(linear.data, label_np.shape[1], label_np.shape[0], bands=1, format="uchar")

    logger.error(f"Writing Label: {label_file}; shape: {label_np.shape}")
    im.write_to_file(
        label_file, pyramid=True, bigtiff=True, tile=True, tile_width=512, tile_height=512, compression="jpeg"
    )

    logger.error(f"Label dimensions: {openslide.OpenSlide(label_file).dimensions}")
    logger.error("Total Time Taken: {:.4f}".format(time.time() - start))


def create_tasks(batch_size, tiles_j, tiles_i, w, h, max_w, max_h):
    coords = []
    infer_tasks = []
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
