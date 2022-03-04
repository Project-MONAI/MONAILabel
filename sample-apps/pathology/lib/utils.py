import logging
import os
import shutil
from math import ceil

import cv2
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

from monailabel.datastore.dsa import DSADatastore

logger = logging.getLogger(__name__)


def split_pannuke_dataset(image, label, output_dir):
    images = np.load(image)
    labels = np.load(label)
    logger.info(f"Image Shape: {images.shape}")
    logger.info(f"Labels Shape: {images.shape}")

    shutil.rmtree(output_dir, True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    dataset_json = []
    for i in tqdm(range(images.shape[0])):
        name = f"img_{str(i).zfill(4)}.npy"
        image_file = os.path.join(images_dir, name)
        label_file = os.path.join(labels_dir, name)

        np.save(image_file, images[i])
        np.save(label_file, labels[i])
        dataset_json.append({"image": image_file, "label": label_file})
    return dataset_json


def split_wsi_image(datastore, d, output_dir, roi_size, groups):
    logger.info(f"Fetching Image/Label : {d}")
    label = d["label"]

    regions_dir = os.path.join(output_dir, "regions")
    tiles_dir = os.path.join(output_dir, "tiles")

    annotations = datastore.get_label(label, "")
    groups = groups if groups else []
    groups = set([g.lower() for g in groups])

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
            if isinstance(datastore, DSADatastore):
                dsa: DSADatastore = datastore

                name = f"{label}_{x}_{y}_{w}_{h}"
                image_path = os.path.realpath(os.path.join(regions_dir, "images", f"{name}.png"))
                os.makedirs(os.path.dirname(image_path), exist_ok=True)

                fetch_from_server = False
                image_uri = datastore.get_image_uri(label)
                if fetch_from_server or not os.path.exists(image_uri):
                    parameters = {
                        "left": x,
                        "top": y,
                        "regionWidth": w,
                        "regionHeight": h,
                        "units": "base_pixels",
                        "encoding": "PNG",
                    }
                    resp = dsa.gc.get(f"/item/{label}/tiles/region", parameters=parameters, jsonResp=False)
                    with open(image_path, "wb") as fp:
                        fp.write(resp.content)
                else:
                    slide = openslide.OpenSlide(datastore.get_image_uri(label))
                    img = slide.read_region((x, y), 0, (w, h)).convert("RGB")
                    img.save(image_path)
                logger.info(f"Image: {image_path}")

                image_np = np.asarray(Image.open(image_path), dtype=np.uint8)
                logger.info(f"Image NP: {image_np.shape}; sum: {np.sum(image_np)}")
                region_to_tiles(name, w, h, image_np, roi_size, os.path.join(tiles_dir, "images"))

                for group, contours in polygons.items():
                    label_np = np.zeros((h, w), dtype=np.uint8)  # Transposed
                    contours = [np.array([[p[0] - x, p[1] - y] for p in contour]) for contour in contours]
                    color = (255, 255, 255)
                    cv2.fillPoly(label_np, pts=contours, color=color)

                    name = f"{label}_{x}_{y}_{w}_{h}_{group}"
                    label_path = os.path.realpath(os.path.join(regions_dir, "labels", f"{name}.png"))
                    os.makedirs(os.path.dirname(label_path), exist_ok=True)
                    cv2.imwrite(label_path, label_np)
                    logger.info(f"Label: {label_path}")
                    region_to_tiles(name, w, h, label_np, roi_size, os.path.join(tiles_dir, "labels"))


def region_to_tiles(name, w, h, input_np, tile_size, output):
    max_w = tile_size[0]
    max_h = tile_size[1]

    tiles_i = ceil(w / max_w)  # COL
    tiles_j = ceil(h / max_h)  # ROW

    logger.info(f"Total Patches to save: {tiles_i * tiles_j}")
    os.makedirs(output, exist_ok=True)

    for tj in range(tiles_j):
        for ti in range(tiles_i):
            tw = min(max_w, w - ti * max_w)
            th = min(max_h, h - tj * max_h)

            sx = ti * max_w
            sy = tj * max_h

            logger.info(f"Patch/Slice ({tj}, {ti}) => {sx}:{sx + tw}, {sy}:{sy + th}")
            if len(input_np.shape) == 3:
                region_rgb = input_np[sy: (sy + th), sx: (sx + tw), :]
                image_np = np.zeros((max_h, max_w, 3), dtype=input_np.dtype)
            else:
                region_rgb = input_np[sy: (sy + th), sx: (sx + tw)]
                image_np = np.zeros((max_h, max_w), dtype=input_np.dtype)
            image_np[0:region_rgb.shape[0], 0:region_rgb.shape[1]] = region_rgb

            logger.info(f"Patch/Slice ({tj}, {ti}) => Size: {region_rgb.shape} / {input_np.shape}")
            img = Image.fromarray(image_np.astype(np.uint8), "RGB" if len(input_np.shape) == 3 else None)
            img.save(os.path.join(output, f"{name}_{tj}x{ti}.png"))
    logger.info(f"Patch(s) Saved...")


def split_dataset(datastore, cache_dir, source, roi_size=(256, 256), groups=None):
    ds = datastore.datalist()

    # PanNuke Dataset
    if source == "pannuke":
        image = np.load(ds[0]["image"]) if len(ds) else None
        if image is not None and len(image.shape) > 3:
            ds = split_pannuke_dataset(ds[0]["image"], ds[0]["label"], cache_dir)

    # WSI (DSA) create patches of roi_size over wsi image+annotation
    if source == "wsi":
        logger.info(f"Split data based on roi: {roi_size}")
        for d in ds:
            split_wsi_image(datastore, d, cache_dir, roi_size, groups)

    logger.info("+++ Total Records: {}".format(len(ds)))
    return ds


def main():
    from monailabel.datastore.dsa import DSADatastore
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    api_url = "http://0.0.0.0:8080/api/v1"
    folder = "621e94e2b6881a7a4bef5170"
    annotation_groups = ["Nuclei"]
    asset_store_path = "/localhome/sachi/Projects/digital_slide_archive/devops/dsa/assetstore"
    api_key = "OJDE9hjuOIS6R8oEqhnVYHUpRpk18NfJABMt36dJ"

    # api_url = "https://demo.kitware.com/histomicstk/api/v1"
    # folder = "5bbdeba3e629140048d017bb"
    # annotation_groups = ["mostly_tumor"]
    # asset_store_path = None
    # api_key = None

    datastore = DSADatastore(api_url, folder, api_key, annotation_groups, asset_store_path)
    ds = datastore.datalist()

    print(f"Dataset for Training: \n{json.dumps(ds, indent=2)}")
    split_dataset(datastore, "/localhome/sachi/Downloads/cache", "wsi", (256, 256), ["Nuclei"])

    # http://0.0.0.0:8080/api/v1/item/621e9513b6881a7a4bef517d/tiles/region
    # parameters = {
    #     "left": 6674,
    #     "top": 22449,
    #     "regionWidth": 1038,
    #     "regionHeight": 616,
    #     "units": "base_pixels",
    #     "exact": False,
    #     "encoding": "JPEG",
    #     "jpegQuality": 95,
    #     "jpegSubsampling": 0,
    # }


if __name__ == "__main__":
    main()
