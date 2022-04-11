import logging
import os
import random
import shutil
from io import BytesIO
from math import ceil

import cv2
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

from monailabel.datastore.dsa import DSADatastore

logger = logging.getLogger(__name__)


def split_pannuke_dataset(image, label, output_dir, groups):
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
        name = f"img_{str(i).zfill(4)}.npy"
        image_file = os.path.join(images_dir, name)
        label_file = os.path.join(labels_dir, name)

        image_np = images[i]
        mask = labels[i]
        label_np = np.zeros(shape=mask.shape[:2])

        for idx, name in label_channels.items():
            if idx < mask.shape[2]:
                m = mask[:, :, idx]
                if np.count_nonzero(m):
                    m[m > 0] = groups.get(name, 1)
                    label_np = np.where(m > 0, m, label_np)

        np.save(image_file, image_np)
        np.save(label_file, label_np)
        dataset_json.append({"image": image_file, "label": label_file})
    return dataset_json


def split_dsa_dataset(datastore, d, output_dir, groups, tile_size, max_region=(10240, 10240)):
    groups = groups if groups else dict()
    groups = [groups] if isinstance(groups, str) else groups
    if not isinstance(groups, dict):
        groups = {v: k + 1 for k, v in enumerate(groups)}
    groups = {k.lower(): v for k, v in groups.items()}
    logger.info(f"++ Using Groups: {groups}")

    logger.info(f"Fetching Image/Label : {d}")
    item_id = d["label"]

    os.makedirs(output_dir, exist_ok=True)

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

        if points:
            logger.info(f"Total Points: {len(points)}")
            x, y, w, h = cv2.boundingRect(np.array(points))
            logger.info(f"ID: {annotation['_id']} => Groups: {polygons.keys()}; Location: ({x}, {y}); Size: {w} x {h}")

            if w > max_region[0]:
                logger.warning(f"Reducing Region to Max-Width; w: {w}; max_w: {max_region[0]}")
                w = max_region[0]
            if h > max_region[1]:
                logger.warning(f"Reducing Region to Max-Height; h: {w}; max_h: {max_region[1]}")
                h = max_region[1]

            if isinstance(datastore, DSADatastore):
                dsa: DSADatastore = datastore
                name = f"{item_id}_{x}_{y}_{w}_{h}"

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
                    resp = dsa.gc.get(f"/item/{item_id}/tiles/region", parameters=parameters, jsonResp=False)
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                else:
                    slide = openslide.OpenSlide(datastore.get_image_uri(item_id))
                    img = slide.read_region((x, y), 0, (w, h)).convert("RGB")

                # image_path = os.path.realpath(os.path.join(regions_dir, f"{name}.png"))
                # os.makedirs(os.path.dirname(image_path), exist_ok=True)
                # img.save(image_path)

                image_np = np.asarray(img, dtype=np.uint8)
                logger.debug(f"Image NP: {image_np.shape}; sum: {np.sum(image_np)}")
                tiled_images = region_to_tiles(name, w, h, image_np, tile_size, output_dir, "Image")

                label_np = np.zeros((h, w), dtype=np.uint8)  # Transposed
                for group, contours in polygons.items():
                    color = groups.get(group, 1)
                    contours = [np.array([[p[0] - x, p[1] - y] for p in contour]) for contour in contours]

                    cv2.fillPoly(label_np, pts=contours, color=color)
                    logger.info(
                        f"{group} => p: {len(contours)}; c: {color}; unique: {np.unique(label_np, return_counts=True)}"
                    )

                    # name = f"{label}_{x}_{y}_{w}_{h}"
                    # label_path = os.path.realpath(os.path.join(regions_dir, "labels", group, f"{name}.png"))
                    # os.makedirs(os.path.dirname(label_path), exist_ok=True)
                    # cv2.imwrite(label_path, label_np)

                tiled_labels = region_to_tiles(
                    name, w, h, label_np, tile_size, os.path.join(output_dir, "labels", "final"), "Label"
                )
                for k in tiled_images:
                    dataset_json.append({"image": tiled_images[k], "label": tiled_labels[k]})
    return dataset_json


def region_to_tiles(name, w, h, input_np, tile_size, output, prefix):
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


def split_dataset(datastore, cache_dir, source, groups, tile_size, max_region=(10240, 10240), limit=0, randomize=True):
    ds = datastore.datalist()
    shutil.rmtree(cache_dir, ignore_errors=True)

    if isinstance(datastore, DSADatastore):
        logger.info(f"DSA:: Split data based on tile size: {tile_size}; groups: {groups}")
        ds_new = []
        count = 0
        if limit > 0:
            ds = random.sample(ds, limit) if randomize else ds[:limit]
        for d in tqdm(ds):
            ds_new.extend(split_dsa_dataset(datastore, d, cache_dir, groups, tile_size, max_region))
            count += 1
            if 0 < limit < count:
                break
        ds = ds_new
    elif source == "pannuke":
        image = np.load(ds[0]["image"]) if len(ds) == 1 else None
        if image is not None and len(image.shape) > 3:
            logger.info(f"PANNuke (For Developer Mode only):: Split data; groups: {groups}")
            ds = split_pannuke_dataset(ds[0]["image"], ds[0]["label"], cache_dir, groups)

    logger.info("+++ Total Records: {}".format(len(ds)))
    return ds


def main():
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
    split_dataset(datastore, "/localhome/sachi/Downloads/dsa/mostly_tumor", "wsi", annotation_groups, (256, 256))


def main_nuke():
    from monailabel.datastore.local import LocalDatastore

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    datastore = LocalDatastore("/localhome/sachi/Data/Pathology/PanNuke", extensions=("*.nii.gz", "*.nii", "*.npy"))
    split_dataset(datastore, "/localhome/sachi/Data/Pathology/PanNukeF", "pannuke", "Nuclei", None)


if __name__ == "__main__":
    main()
