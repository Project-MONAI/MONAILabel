import json
import logging
import random
import tempfile
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

label_color_map: Dict[str, Any] = dict()


def get_color(label, color_map, hex=False):
    color = color_map.get(label) if color_map else None
    color = label_color_map.get(label) if not color else color
    if color is None:
        color = [random.randint(0, 255) for _ in range(3)]
        label_color_map[label] = color
    if hex:
        return "#%02x%02x%02x" % tuple(color)
    return "rgb(" + ",".join([str(x) for x in color]) + ")"


def create_dsa_annotations_json(
    json_data, name="monailabel", description="Annotations generated by MONAILabel", color_map=None
):
    labels = set()
    annotation_doc = {"name": name, "description": description, "elements": []}
    for tid, r in json_data["tasks"].items():
        logger.debug(f"Adding annotations for task: {tid}")
        for v in r.get("annotations", []):
            label = v["label"]
            labels.add(label)

            color = get_color(label, color_map)
            logger.debug(f"Adding Contours for label: {label}; color: {color}")

            contours = v["contours"]
            for contour in contours:
                a = np.array(contour)
                points = np.hstack((a, np.zeros((a.shape[0], 1), dtype=a.dtype))).tolist()
                annotation_style = {
                    "group": label,
                    "type": "polyline",
                    "lineColor": color,
                    "lineWidth": 2.0,
                    "closed": True,
                    "points": points,
                    "label": {"value": label},
                }
                annotation_doc["elements"].append(annotation_style)

    logger.debug(f"Total Elements: {len(annotation_doc['elements'])}")

    label_json = tempfile.NamedTemporaryFile(suffix=".json").name
    with open(label_json, "w") as fp:
        json.dump(annotation_doc, fp)
    return label_json


def create_asap_annotations_xml(json_data, color_map=None):
    total_count = 0
    label_xml = tempfile.NamedTemporaryFile(suffix=".xml").name

    with open(label_xml, "w") as fp:
        fp.write('<?xml version="1.0"?>\n')
        fp.write("<ASAP_Annotations>\n")
        fp.write("  <Annotations>\n")

        labels = set()
        for tid, r in json_data["tasks"].items():
            logger.debug(f"Adding annotations for task: {tid}")
            for v in r.get("annotations", []):
                label = v["label"]
                labels.add(label)

                color = get_color(label, color_map)
                logger.debug(f"Adding Contours for label: {label}; color: {color}")

                contours = v["contours"]
                for contour in contours:
                    fp.write(f'    <Annotation Name="{label}" Type="Polygon" PartOfGroup="{label}" Color="{color}">\n')
                    fp.write("      <Coordinates>\n")
                    for pcount, point in enumerate(contour):
                        fp.write('        <Coordinate Order="{}" X="{}" Y="{}" />\n'.format(pcount, point[0], point[1]))
                    fp.write("      </Coordinates>\n")
                    fp.write("    </Annotation>\n")
                    total_count += 1
        fp.write("  </Annotations>\n")
        fp.write("  <AnnotationGroups>\n")
        for label in labels:
            fp.write(f'    <Group Name="{label}" PartOfGroup="None" Color="{get_color(label, color_map)}">\n')
            fp.write("      <Attributes />\n")
            fp.write("    </Group>\n")
        fp.write("  </AnnotationGroups>\n")
        fp.write("</ASAP_Annotations>\n")

    logger.debug(f"Total Polygons: {total_count}")
    return label_xml
