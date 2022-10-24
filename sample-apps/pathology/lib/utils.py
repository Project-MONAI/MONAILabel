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

import copy
import logging
import math
import os
import random
import shutil
import xml.etree.ElementTree
from io import BytesIO
from math import ceil

import cv2
import numpy as np
import openslide
from PIL import Image
from skimage.measure import regionprops
from tqdm import tqdm

from monailabel.datastore.dsa import DSADatastore
from monailabel.datastore.local import LocalDatastore
from monailabel.interfaces.datastore import Datastore
from monailabel.utils.others.generic import get_basename, get_basename_no_ext, is_openslide_supported

logger = logging.getLogger(__name__)


def split_dataset(
    datastore: Datastore, cache_dir, source, groups, tile_size, max_region=(10240, 10240), limit=0, randomize=True
):
    ds = datastore.datalist()
    output_dir = cache_dir
    if output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)

    if source == "none":
        pass
    elif source == "pannuke":
        logger.info("Prepare from PanNuke Dataset")

        ds_new = []
        for i in range(len(ds)):
            image = ds[i]["image"]
            label = ds[i]["label"]
            logger.info(f"{image} => PANNuke (For Developer Mode only):: Split data; groups: {groups}")
            d = split_pannuke_dataset(image, label, output_dir, groups)
            logger.info(f"{image} => Total Dataset Records: {len(ds)}")
            ds_new.extend(d)
        ds = ds_new
        logger.info(f"Total Dataset Records: {len(ds)}")
    elif source == "nuclick":
        logger.info("Split data based on each nuclei")

        ds_new = []
        for d in tqdm(ds):
            ds_new.extend(split_nuclei_dataset(d, output_dir))
            if 0 < limit < len(ds_new):
                ds_new = ds_new[:limit]
                break
        ds = ds_new
    else:
        logger.info(f"Split data based on tile size: {tile_size}; groups: {groups}")
        ds_new = []
        count = 0
        if limit > 0:
            ds = random.sample(ds, limit) if randomize else ds[:limit]
        for d in tqdm(ds):
            if isinstance(datastore, DSADatastore):
                ds_new.extend(split_dsa_dataset(datastore, d, output_dir, groups, tile_size, max_region))
            else:
                ds_new.extend(split_local_dataset(datastore, d, output_dir, groups, tile_size, max_region))
            count += 1
            if 0 < limit < count:
                break
        ds = ds_new

    logger.info(f"+++ Total Records: {len(ds)}")
    return ds


def split_pannuke_dataset(image, label, output_dir, groups, save_as_png=True):
    groups = groups if groups else dict()
    groups = [groups] if isinstance(groups, str) else groups
    if not isinstance(groups, dict):
        groups = {v: k + 1 for k, v in enumerate(groups)}

    label_channels = {
        0: "Neoplastic cells",
        1: "Inflammatory",
        2: "Connective/Soft tissue cells",
        3: "Dead Cells",
        4: "Epithelial",
    }

    logger.info(f"++ Using Groups: {groups}")
    logger.info(f"++ Using Label Channels: {label_channels}")

    image_id = get_basename_no_ext(image)
    images = np.load(image)
    labels = np.load(label)
    logger.info(f"Image Shape: {images.shape}")
    logger.info(f"Labels Shape: {labels.shape}")

    images_dir = output_dir
    labels_dir = os.path.join(output_dir, "labels", "final")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    dataset_json = []
    for i in tqdm(range(images.shape[0])):
        filename = f"{image_id}_{str(i).zfill(4)}.npy"
        image_np = images[i]
        mask = labels[i]
        label_np = np.zeros(shape=mask.shape[:2])

        for idx, name in label_channels.items():
            if idx < mask.shape[2]:
                m = mask[:, :, idx]
                if np.count_nonzero(m):
                    m[m > 0] = groups.get(name, 1)
                    label_np = np.where(m > 0, m, label_np)

        if save_as_png:
            image_png = Image.fromarray(image_np.astype(np.uint8), "RGB" if len(image_np.shape) == 3 else None)
            label_png = Image.fromarray(label_np.astype(np.uint8), "RGB" if len(label_np.shape) == 3 else None)

            image_file = os.path.join(images_dir, filename.replace(".npy", ".png"))
            label_file = os.path.join(labels_dir, filename.replace(".npy", ".png"))

            image_png.save(image_file)
            label_png.save(label_file)
        else:
            image_file = os.path.join(images_dir, filename)
            label_file = os.path.join(labels_dir, filename)

            np.save(image_file, image_np)
            np.save(label_file, label_np)

        dataset_json.append({"image": image_file, "label": label_file})
    return dataset_json


def split_dsa_dataset(datastore, d, output_dir, groups, tile_size, max_region=(10240, 10240)):
    groups, item_id = _group_item(groups, d, output_dir)
    dataset_json = []

    annotations = datastore.get_label(item_id, "")
    for annotation in annotations:
        points = []
        polygons = {g: [] for g in groups}
        elements = annotation["annotation"]["elements"]
        for e in elements:
            g = e.get("group")
            g = g if g else "None"
            g = g.lower()

            if g in groups:
                p = e["points"]
                p = np.delete(np.array(p), 2, 1).tolist()
                if p:
                    polygons[g].append(p)
                    points.extend(p)

        if not points:
            continue

        x, y, w, h = _to_roi(points, max_region, polygons, annotation["_id"])

        image_uri = datastore.get_image_uri(item_id)
        if not os.path.exists(image_uri):
            parameters = {
                "left": x,
                "top": y,
                "regionWidth": w,
                "regionHeight": h,
                "units": "base_pixels",
                "encoding": "PNG",
            }
            dsa: DSADatastore = datastore
            resp = dsa.gc.get(f"/item/{item_id}/tiles/region", parameters=parameters, jsonResp=False)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            slide = openslide.OpenSlide(datastore.get_image_uri(item_id))
            img = slide.read_region((x, y), 0, (w, h)).convert("RGB")

        dataset_json.extend(_to_dataset(item_id, x, y, w, h, img, tile_size, polygons, groups, output_dir))

    return dataset_json


def split_local_dataset(datastore, d, output_dir, groups, tile_size, max_region=(10240, 10240)):
    groups, item_id = _group_item(groups, d, output_dir)
    local: LocalDatastore = datastore
    item_id = local._to_id(item_id)[0]

    dataset_json = []

    points = []
    polygons = {g: [] for g in groups}

    annotations_xml = xml.etree.ElementTree.parse(d["label"]).getroot()
    for annotation in annotations_xml.iter("Annotation"):
        g = annotation.get("PartOfGroup")
        g = g if g else "None"
        g = g.lower()

        if g not in groups:
            continue

        p = []
        for e in annotation.iter("Coordinate"):
            xy = [int(e.get("X")), int(e.get("Y"))]
            if sum(xy):
                p.append(xy)

        if p:
            polygons[g].append(p)
            points.extend(p)

    x, y, w, h = _to_roi(points, max_region, polygons, item_id)
    if is_openslide_supported(d["image"]):
        slide = openslide.OpenSlide(d["image"])
        img = slide.read_region((x, y), 0, (w, h)).convert("RGB")
    else:
        img = Image.open(d["image"]).convert("RGB")
        w = img.size[0]
        h = img.size[1]

    dataset_json.extend(_to_dataset(item_id, x, y, w, h, img, tile_size, polygons, groups, output_dir))
    return dataset_json


def split_nuclei_dataset(
    d,
    output_dir,
    centroid_key="centroid",
    nuclei_id_key="nuclei_id",
    class_key="class",
    min_area=80,
    min_distance=20,
    crop_size=128,
):
    dataset_json = []
    ignored = 0

    images_dir = output_dir
    labels_dir = os.path.join(output_dir, "labels", "final")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image = Image.open(d["image"])
    image_np = np.array(image)
    image_id = get_basename_no_ext(d["image"])

    mask = Image.open(d["label"])
    mask_np = np.array(mask)

    nuclei_id = 1
    for label_idx in np.unique(mask_np):
        if label_idx == 0:
            continue

        label_np = np.where(mask_np == label_idx, label_idx, 0).astype(np.uint8)
        _, labels, _, _ = cv2.connectedComponentsWithStats(label_np, 4, cv2.CV_32S)

        stats = regionprops(labels)
        for stat in stats:
            x, y = stat.centroid
            x = int(math.floor(x))
            y = int(math.floor(y))

            if stat.area < min_area:
                logger.debug(f"++++ Ignored label-{label_idx} with smaller area => ( {stat.area} < {min_area})")
                ignored += 1
                continue

            if (
                x < min_distance
                or y < min_distance
                or (image.size[-2] - x) < min_distance
                or (image.size[-1] - y < min_distance)
            ):
                logger.debug(
                    f"++++ Ignored label-{label_idx} with close to boundary edge => ({x},{y}) < {min_distance} in {image.size}"
                )
                ignored += 1
                continue

            if label_np[x][y] == 0:
                logger.debug(
                    f"++++ Ignored label-{label_idx} with centroid falling over background => ({x},{y}) => {label_idx})"
                )
                ignored += 1
                continue

            item = copy.deepcopy(d)
            item[centroid_key] = (x, y)
            item[nuclei_id_key] = nuclei_id
            item[class_key] = label_idx
            item["image_path"] = item["image"]

            bbox = compute_bbox(crop_size, (x, y), image.size)
            cropped_image_np = image_np[bbox[0] : bbox[2], bbox[1] : bbox[3], :]
            cropped_image = Image.fromarray(cropped_image_np, "RGB")

            this_label = np.where(labels == stat.label, label_idx, 0).astype(np.uint8)
            cropped_label_np = this_label[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            cropped_label = Image.fromarray(cropped_label_np.astype(np.uint8), None)

            filename = f"{image_id}_{label_idx}_{str(nuclei_id).zfill(4)}.npy"
            image_file = os.path.join(images_dir, filename.replace(".npy", ".png"))
            label_file = os.path.join(labels_dir, filename.replace(".npy", ".png"))

            cropped_image.save(image_file)
            cropped_label.save(label_file)

            # logger.info(f"{d['label']} => {len(stats)} => {mask.shape} => {stat.label} => {cval}")
            item["image"] = image_file
            item["label"] = label_file
            dataset_json.append(item)
            nuclei_id += 1

    if ignored:
        logger.debug(f"Total Ignored => {d['label']} => {ignored}/{len(stats)}")
    return dataset_json


def compute_bbox(patch_size, centroid, size):
    x, y = centroid
    m, n = size

    x_start = int(max(x - patch_size / 2, 0))
    y_start = int(max(y - patch_size / 2, 0))
    x_end = x_start + patch_size
    y_end = y_start + patch_size
    if x_end > m:
        x_end = m
        x_start = m - patch_size
    if y_end > n:
        y_end = n
        y_start = n - patch_size
    return x_start, y_start, x_end, y_end


def _group_item(groups, d, output_dir):
    groups = groups if groups else dict()
    groups = [groups] if isinstance(groups, str) else groups
    if not isinstance(groups, dict):
        groups = {v: k + 1 for k, v in enumerate(groups)}
    groups = {k.lower(): v for k, v in groups.items()}
    logger.info(f"++ Using Groups: {groups}")

    logger.info(f"Fetching Image/Label : {d}")
    item_id = get_basename(d["label"])

    os.makedirs(output_dir, exist_ok=True)
    return groups, item_id


def _to_roi(points, max_region, polygons, annotation_id):
    logger.info(f"Total Points: {len(points)}")
    x, y, w, h = cv2.boundingRect(np.array(points))
    logger.info(f"ID: {annotation_id} => Groups: {polygons.keys()}; Location: ({x}, {y}); Size: {w} x {h}")

    if w > max_region[0]:
        logger.warning(f"Reducing Region to Max-Width; w: {w}; max_w: {max_region[0]}")
        w = max_region[0]
    if h > max_region[1]:
        logger.warning(f"Reducing Region to Max-Height; h: {h}; max_h: {max_region[1]}")
        h = max_region[1]
    return x, y, w, h


def _to_dataset(item_id, x, y, w, h, img, tile_size, polygons, groups, output_dir, debug=False):
    dataset_json = []

    name = f"{item_id}_{x}_{y}_{w}_{h}"
    if debug:
        regions_dir = os.path.join(output_dir, "regions")
        image_path = os.path.realpath(os.path.join(regions_dir, f"{name}.png"))
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img.save(image_path)

    image_np = np.asarray(img, dtype=np.uint8)
    logger.debug(f"Image NP: {image_np.shape}; sum: {np.sum(image_np)}")
    tiled_images = _region_to_tiles(name, w, h, image_np, tile_size, output_dir, "Image")

    label_np = np.zeros((h, w), dtype=np.uint8)  # Transposed
    for group, contours in polygons.items():
        color = groups.get(group, 1)
        contours = [np.array([[p[0] - x, p[1] - y] for p in contour]) for contour in contours]

        cv2.fillPoly(label_np, pts=contours, color=color)
        logger.info(f"{group} => p: {len(contours)}; c: {color}; unique: {np.unique(label_np, return_counts=True)}")

        if debug:
            regions_dir = os.path.join(output_dir, "regions")
            label_path = os.path.realpath(os.path.join(regions_dir, "labels", group, f"{name}.png"))
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            cv2.imwrite(label_path, label_np)

    tiled_labels = _region_to_tiles(
        name, w, h, label_np, tile_size, os.path.join(output_dir, "labels", "final"), "Label"
    )
    for k in tiled_images:
        dataset_json.append({"image": tiled_images[k], "label": tiled_labels[k]})
    return dataset_json


def _region_to_tiles(name, w, h, input_np, tile_size, output, prefix):
    max_w = tile_size[0]
    max_h = tile_size[1]

    tiles_i = ceil(w / max_w)  # COL
    tiles_j = ceil(h / max_h)  # ROW

    logger.info(f"{prefix} => Input: {input_np.shape}; Total Patches to save: {tiles_i * tiles_j}")
    os.makedirs(output, exist_ok=True)

    result = {}
    for tj in range(tiles_j):
        for ti in range(tiles_i):
            tw = min(max_w, w - ti * max_w)
            th = min(max_h, h - tj * max_h)

            sx = ti * max_w
            sy = tj * max_h

            logger.debug(f"{prefix} => Patch/Slice ({tj}, {ti}) => {sx}:{sx + tw}, {sy}:{sy + th}")
            if len(input_np.shape) == 3:
                region_rgb = input_np[sy : (sy + th), sx : (sx + tw), :]
                image_np = np.zeros((max_h, max_w, 3), dtype=input_np.dtype)
            else:
                region_rgb = input_np[sy : (sy + th), sx : (sx + tw)]
                image_np = np.zeros((max_h, max_w), dtype=input_np.dtype)
            image_np[0 : region_rgb.shape[0], 0 : region_rgb.shape[1]] = region_rgb

            logger.debug(f"{prefix} => Patch/Slice ({tj}, {ti}) => Size: {region_rgb.shape} / {input_np.shape}")
            img = Image.fromarray(image_np.astype(np.uint8), "RGB" if len(input_np.shape) == 3 else None)

            filename = f"{name}_{tj}x{ti}.png"
            save_path = os.path.join(output, filename)
            img.save(save_path)
            result[filename] = save_path
    logger.debug(f"{prefix} => Total {len(result)} Patch(s) are Saved at: {output}")
    return result


def main_dsa():
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # api_url = "http://0.0.0.0:8080/api/v1"
    # folder = "621e94e2b6881a7a4bef5170"
    # annotation_groups = ["Nuclei"]
    # asset_store_path = "/localhome/sachi/Projects/digital_slide_archive/devops/dsa/assetstore"
    # api_key = "OJDE9hjuOIS6R8oEqhnVYHUpRpk18NfJABMt36dJ"

    api_url = "https://demo.kitware.com/histomicstk/api/v1"
    folder = "5bbdeba3e629140048d017bb"
    annotation_groups = ["mostly_tumor"]
    asset_store_path = None
    api_key = None

    datastore = DSADatastore(api_url, folder, api_key, annotation_groups, asset_store_path)
    print(json.dumps(datastore.datalist(), indent=2))
    split_dataset(datastore, "/localhome/sachi/Downloads/dsa/mostly_tumor", "", annotation_groups, (256, 256))


def main_nuke():
    from monailabel.datastore.local import LocalDatastore

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    datastore = LocalDatastore("/localhome/sachi/Datasets/pannuke", extensions=("*.npy"))
    labels = {
        "Neoplastic cells": 1,
        "Inflammatory": 2,
        "Connective/Soft tissue cells": 3,
        "Dead Cells": 4,
        "Epithelial": 5,
    }
    split_dataset(datastore, "/localhome/sachi/Datasets/pannukeF", "pannuke", labels, None)


def main_local():
    import json

    from monailabel.datastore.local import LocalDatastore

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    annotation_groups = ["Nuclei"]
    datastore = LocalDatastore("C:\\Projects\\Pathology\\Test", extensions=("*.svs", "*.xml"))
    print(json.dumps(datastore.datalist(), indent=2))

    split_dataset(datastore, "C:\\Projects\\Pathology\\TestF", "", annotation_groups, (256, 256))
    # print(json.dumps(ds, indent=2))


def main_nuclei():
    from pathlib import Path

    from monailabel.datastore.local import LocalDatastore

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    home = str(Path.home())
    studies = f"{home}/Dataset/Pathology/pannukeF"
    output_dir = f"{home}/Dataset/Pathology/pannukeFFF"

    datastore = LocalDatastore(studies, extensions=("*.png", "*.npy"))
    split_dataset(datastore, output_dir, "nuclick", None, None, limit=0)


if __name__ == "__main__":
    main_nuclei()
