import logging
import os
import shutil

import numpy as np
from tqdm import tqdm

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


def split_wsi_image(datastore, d, output_dir, roi_size):
    pass


def split_dataset(datastore, cache_dir, source, roi_size=(256, 256)):
    ds = datastore.datalist()

    # PanNuke Dataset
    if source == "pannuke":
        image = np.load(ds[0]["image"]) if len(ds) else None
        if image is not None and len(image.shape) > 3:
            ds = split_pannuke_dataset(ds[0]["image"], ds[0]["label"], cache_dir)

    # WSI (DSA) create patches of roi_size over wsi image+annotation
    if source == "wsi":
        logger.info(f"Split data based on roi: {roi_size}")
        for d in tqdm(ds):
            split_wsi_image(datastore, d, cache_dir, roi_size)

    logger.info("+++ Total Records: {}".format(len(ds)))
    return ds
