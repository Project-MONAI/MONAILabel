import argparse
import glob
import json
import logging
import os
import xml.etree.cElementTree
from math import ceil
from multiprocessing import Pool

import cv2
import numpy as np
import openslide
from skimage.measure import points_in_poly

from monailabel.utils.others.generic import get_basename, file_ext


def fetch_annotations(image, label, coverage, min_size, level=0):
    tree = xml.etree.cElementTree.ElementTree(file=label)

    name = get_basename(label)
    name = name.replace(file_ext(name), "")
    logger = logging.getLogger(name)

    annotations = []

    slide = openslide.OpenSlide(image)
    factor = slide.level_downsamples[level]

    logger.info(f"Using Factor: {factor} => Level: {level}")

    for annotation in tree.getroot().iter('Annotation'):
        group = int(
            annotation.attrib.get(
                "PartOfGroup").replace(
                "_", "").replace(
                "Tumor", "0").replace(
                "Exclusion", "2").replace(
                "None", "1"))
        object_type = annotation.attrib.get("Type").lower()
        idx = annotation.attrib.get("Name").lstrip("_")

        for coords in annotation.iter('Coordinates'):
            points = []
            for coord in coords:
                points.append(
                    [round(float(coord.attrib.get("X")) / factor), round(float(coord.attrib.get("Y")) / factor)])

            x, y, w, h = cv2.boundingRect(np.array(points))
            center_x = round(x + w / 2)
            center_y = round(y + h / 2)

            if ((w * h) / min_size[0] * min_size[1]) > 100:
                coverage = 1.2

            width = round(max(min_size[0], w * coverage))
            height = round(max(min_size[0], h * coverage))
            width = round(min_size[0] * ceil(width / min_size[0]))
            height = round(min_size[1] * ceil(height / min_size[1]))

            loc_x = max(0, round(center_x - width / 2))
            loc_y = max(0, round(center_y - height / 2))
            logger.info(f"{idx} => tumor size: {w} x {h} => region: ({loc_x}, {loc_y}) => {width} x {height}")

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

    # Make sure all Name defined by annotator are unique from the given xml
    ids = [a["idx"] for a in annotations]
    uids = set(ids)
    assert len(ids) == len(uids)

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
    a = [a for a in annotations if a["idx"] == idx][0]
    if not a["ignore"]:
        return idx
    return get_referer(a["refer"], annotations)


def get_referred_by(idx, annotations):
    return [a["idx"] for a in annotations if a["idx"] == idx or a["refer"] == idx]


def get_matching(idx, annotations):
    for a in annotations:
        if a["idx"] == idx:
            return a
    return None


def create_region_image(annotations, output, level=0, output_ext=".png"):
    for annotation in annotations:
        if annotation["ignore"]:
            continue

        name = annotation["name"]
        idx = annotation["idx"]
        image = annotation["image"]

        logger = logging.getLogger(f"{name}_{idx}")
        logger.info(f"Create Image; bbox: {annotation['bbox']}; region: {annotation['region_size']}")

        slide = openslide.OpenSlide(image)
        save_patch(annotation, None, slide, level, output, output_ext)


def save_patch(annotation, input_np, slide, level, output, output_ext, max_w=4096, max_h=4096):
    name = annotation["name"]
    idx = annotation["idx"]
    logger = logging.getLogger(f"{name}_{idx}")

    w, h = annotation["region_size"]
    x, y = annotation["region_top"]
    tiles_i = ceil(w / max_w)  # COL
    tiles_j = ceil(h / max_h)  # ROW

    prefix = "Image" if slide else "Label"

    logger.info(f"{prefix} - Total Patches to save: {tiles_i * tiles_j}")
    for tj in range(tiles_j):
        for ti in range(tiles_i):
            tw = min(max_w, w - ti * max_w)
            th = min(max_h, h - tj * max_h)

            if slide:
                tx = x + ti * max_w
                ty = y + tj * max_h

                logger.info(f"{prefix} - Patch/Slide ({tj}, {ti}) => Top: ({tx}, {ty}); Size: {tw} x {th}")
                region_rgb = slide.read_region((tx, ty), level, (tw, th)).convert("RGB")
                region_rgb.save(os.path.join(output, f"{name}_{idx}_{tj}x{ti}{output_ext}"))
            else:
                sx = ti * max_w
                sy = tj * max_h

                logger.info(f"{prefix} - Patch/Slice ({tj}, {ti}) => {sx}:{sx + tw}, {sy}:{sy + th}")
                region_rgb = input_np[sy:(sy + th), sx:(sx + tw)]
                logger.info(f"{prefix} - Patch/Slice ({tj}, {ti}) => Size: {region_rgb.shape} / {input_np.shape}")
                cv2.imwrite(os.path.join(output, f"{name}_{idx}_{tj}x{ti}{output_ext}"), region_rgb)
    logger.info(f"{prefix} Patch(s) Saved...")


def create_region_label(annotations, output, output_ext=".png"):
    for annotation in annotations:
        if annotation["ignore"]:
            continue

        # save label and all references
        idx = annotation["idx"]
        name = annotation["name"]
        referred_by = get_referred_by(idx, annotations)

        logger = logging.getLogger(f"{name}_{idx}")
        logger.info(f"Create Label; region: {annotation['region_size']}; bbox: {annotation['bbox']}; By: {referred_by}")

        x, y, w, h = annotation["bbox"]
        loc_x, loc_y = annotation["region_top"]
        width, height = annotation["region_size"]

        label_np = np.zeros((height, width), dtype=np.uint8)  # Transposed
        logger.info("Label NP Created!!")

        for r in referred_by:
            ra = get_matching(r, annotations)
            annotation_points = ra["points"]
            annotation_group = ra["group"]

            logger.info(f"Adding Label For Index: {r}; size: {width} x {height}; points: {len(annotation_points)}")

            contours = np.array([[p[0] - loc_x, p[1] - loc_y] for p in annotation_points])
            color = (255, 255, 255) if annotation_group == 0 else (128, 128, 128) if annotation_group == 1 else (
                0, 0, 0)
            cv2.fillPoly(label_np, pts=[contours], color=color)

        logger.info(f"Label Ready...")
        save_patch(annotation, label_np, None, 0, output, output_ext)


def create_region(annotations, output, level, output_ext=".png"):
    #create_region_image(annotations, os.path.join(output, "images"), level, output_ext)
    create_region_label(annotations, os.path.join(output, "labels"), output_ext)


def run_job(job):
    annotations = job["annotations"]
    args = job["args"]
    create_region(annotations, args.output, args.level, args.extension)


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
    parser.add_argument("-n", "--level", type=int, default=0)
    parser.add_argument("-c", "--coverage", type=float, default=2.0)
    parser.add_argument("-s", "--min_size", default="[4096,4096]")
    parser.add_argument("-x", "--extension", default=".png")
    parser.add_argument("-m", "--multiprcoess", type=bool, default=True)

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
            "annotations": fetch_annotations(image, label, args.coverage, args.min_size, args.level),
            "args": args,
        }

    if args.multiprcoess:
        with Pool(processes=16) as pool:
            pool.map(run_job, annotations.values())
    else:
        for a in annotations.values():
            run_job(a)


if __name__ == "__main__":
    main()
