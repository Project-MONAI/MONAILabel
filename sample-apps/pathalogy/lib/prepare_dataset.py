import argparse
import glob
import json
import logging
import os
import xml.etree.cElementTree
from multiprocessing import Pool

import cv2
import numpy as np
import openslide
from skimage.measure import points_in_poly

from monailabel.utils.others.generic import get_basename, file_ext


def fetch_annotations(image, label, coverage, min_size, factor=1.0):
    tree = xml.etree.cElementTree.ElementTree(file=label)

    name = get_basename(label)
    name = name.replace(file_ext(name), "")
    logger = logging.getLogger(name)

    annotations = []
    idx = 0
    for annotation in tree.getroot().iter('Annotation'):
        group = int(annotation.attrib.get("PartOfGroup").replace("_", ""))
        object_type = annotation.attrib.get("Type").lower()

        for coords in annotation.iter('Coordinates'):
            points = []
            for coord in coords:
                points.append([int(float(coord.attrib.get("X")) / factor), int(float(coord.attrib.get("Y")) / factor)])

            x, y, w, h = cv2.boundingRect(np.array(points))
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            width = int(max(min_size[0], w * coverage))
            height = int(max(min_size[0], h * coverage))
            loc_x = max(0, int(center_x - width / 2))
            loc_y = max(0, int(center_y - height / 2))
            logger.info(f"tumor size: {w} x {h} => region size: {width} x {height}")

            annotations.append({
                "name": name,
                "image": image,
                "label": label,
                "idx": idx,
                "points": points,
                "group": group,
                "type": object_type,
                "bbox": (x, y, w, h),
                "region_top": (loc_x, loc_y),
                "region_size": (width, height),
                "ignore": False,
                "refer": None,
            })
            idx += 1

    while dedup(annotations):
        logger.info("Run Dedup Again...! Dedup Count:{}".format(len([a for a in annotations if a["ignore"]])))

    logger.info("{} => Dedup/Total: {}/{}".format(name, len([a for a in annotations if a["ignore"]]), len(annotations)))
    logger.info("{} => {}".format(name, [a["idx"] for a in annotations if not a["ignore"]]))

    # Fix Chained Refer
    for annotation in annotations:
        if annotation["ignore"]:
            annotation["refer"] = get_referer(annotation["idx"], annotations)
    return annotations


def check_part_of(a, r):
    logger = logging.getLogger(a["name"])

    x, y = r["region_top"]
    w, h = r["region_size"]
    poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    overlap = np.array(points_in_poly(a["points"], poly))
    if np.all(overlap):
        logger.info(f"{a['label']} => {a['idx']} is part of {r['idx']}")
        a["ignore"] = True
        a["refer"] = r["idx"]
        return True
    elif np.any(overlap):
        logger.warning(f"{a['label']} => {a['idx']} partial overlaps with {r['idx']}")
    return False


def dedup(annotations):
    ignored = 0
    for i in range(1, len(annotations)):
        if annotations[i]["ignore"]:
            continue

        for j in range(i):
            if annotations[j]["ignore"]:
                continue
            if check_part_of(annotations[i], annotations[j]):  # I is part of J
                ignored += 1
            elif check_part_of(annotations[j], annotations[i]):  # J is part of I
                ignored += 1
    return ignored


def get_referer(idx, annotations):
    a = annotations[idx]
    if not a["ignore"]:
        return idx
    return get_referer(a["refer"], annotations)


def get_referred_by(idx, annotations):
    return [a["idx"] for a in annotations if a["idx"] == idx or a["refer"] == idx]


def create_region_image(annotations, output, output_ext=".png"):
    for annotation in annotations:
        if annotation["ignore"]:
            continue

        name = annotation["name"]
        idx = annotation["idx"]
        image = annotation["image"]

        logger = logging.getLogger(f"{name}_{idx}")
        logger.info(f"Starting to create region image...")

        slide = openslide.OpenSlide(image)
        region_rgb = slide.read_region(annotation["region_top"], 0, annotation["region_size"]).convert('RGB')
        logger.info(f"Image Ready...")

        region_rgb.save(os.path.join(output, f"{name}_{idx}{output_ext}"))
        logger.info(f"Image Saved...")


def create_region_label(annotations, output, output_ext=".png"):
    for annotation in annotations:
        if annotation["ignore"]:
            continue

        # save label and all references
        idx = annotation["idx"]
        name = annotation["name"]
        referred_by = get_referred_by(idx, annotations)

        logger = logging.getLogger(f"{name}_{idx}")
        logger.info(f"Starting to create region label... Label Idx: {referred_by}")

        x, y, w, h = annotation["bbox"]
        loc_x, loc_y = annotation["region_top"]
        width, height = annotation["region_size"]

        label_np = np.zeros((height, width), dtype=np.uint8)  # Transposed

        for r in referred_by:
            logger.info(f"Adding Label For Index: {r}")
            annotation_points = annotations[r]["points"]
            annotation_group = annotations[r]["group"]
            annotation_type = annotations[r]["type"]

            if annotation_type == "polygon":
                contours = np.array([[p[0] - loc_x, p[1] - loc_y] for p in annotation_points])
                color = (255, 255, 255) if annotation_group in (0, 1) else (0, 0, 0)
                cv2.fillPoly(label_np, pts=[contours], color=color)
            else:
                # TODO:: spline
                for x_idx in range(w):
                    xn = x - loc_x + x_idx
                    for y_idx in range(h):
                        yn = abs(loc_y - y) + y_idx  # y = height from bottom
                        point = (x + x_idx, y + y_idx)
                        if point in annotation_points:
                            label_np[yn][xn] = annotation_group + 1 if annotation_group in (0, 1) else 0

        logger.info(f"Label Ready... sum: {np.sum(label_np)}")

        cv2.imwrite(os.path.join(output, f"{name}_{idx}{output_ext}"), label_np)
        logger.info("Label Saved...")


def create_region(annotations, output, output_ext=".png"):
    create_region_image(annotations, os.path.join(output, "images"), output_ext)
    create_region_label(annotations, os.path.join(output, "labels"), output_ext)


def run_job(job):
    annotations = job["annotations"]
    args = job["args"]
    create_region(annotations, args.output, args.extension)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_dir = "/local/sachi/Data/Pathology/Camelyon"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=f"{root_dir}/79397/training/images/tumor/*.tif")
    parser.add_argument("-l", "--label", default=f"{root_dir}/79397/training/images/tumor/*.xml")
    parser.add_argument("-o", "--output", default=f"{root_dir}/dataset/training/")
    parser.add_argument("-c", "--coverage", type=float, default=3.0)
    parser.add_argument("-s", "--min_size", default="[512,512]")
    parser.add_argument("-x", "--extension", default=".png")

    args = parser.parse_args()
    args.min_size = json.loads(args.min_size)

    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "labels"), exist_ok=True)

    images = sorted(glob.glob(args.image))
    labels = sorted(glob.glob(args.label))

    annotations = {}
    for image, label in zip(images, labels):
        assert os.path.exists(image)
        assert os.path.exists(label)
        annotations[image] = {
            "annotations": fetch_annotations(image, label, args.coverage, args.min_size),
            "args": args,
        }

    with Pool() as pool:
        pool.map(run_job, annotations.values())


if __name__ == "__main__":
    main()
