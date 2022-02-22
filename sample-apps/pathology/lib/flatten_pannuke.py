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
    images_dir = output_dir
    labels_dir = os.path.join(images_dir, "labels", "final")
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


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", default="/local/sachi/Data/Pathology/PanNuke/Fold 1/images/fold1/images.npy")
    parser.add_argument("-m", "--masks", default="/local/sachi/Data/Pathology/PanNuke/Fold 1/masks/fold1/masks.npy")
    parser.add_argument("-o", "--output", default="/local/sachi/Data/Pathology/PanNukeF")

    args = parser.parse_args()
    split_pannuke_dataset(args.images, args.masks, args.output)
    logger.info(f"Dataset Ready: {args.output}")


if __name__ == "__main__":
    main()
