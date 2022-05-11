import json
import logging
import tempfile

import numpy as np

from monailabel.utils.others.label_colors import to_hex, to_rgb

logger = logging.getLogger(__name__)


def create_dsa_annotations_json_old(json_data, loglevel="INFO"):
    logger.setLevel(loglevel.upper())

    annotation_doc = {
        "name": json_data["name"],
        "description": json_data["description"],
        "model": json_data["model"],
        "location": json_data["location"],
        "size": json_data["size"],
        "elements": [],
    }

    for tid, res in enumerate(json_data["annotations"]):
        logger.debug(f"Adding annotations for tile: {tid}")
        if not res:
            continue

        annotation = res.get("annotation")
        if not annotation:
            continue

        color_map = annotation.get("labels")
        elements = annotation.get("elements", [])
        for element in elements:
            label = element["label"]
            color = to_rgb(color_map.get(label))

            logger.info(f"Adding Contours for label: {label}; color: {color}; color_map: {color_map}")

            contours = element["contours"]
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

    logger.info(f"Total Elements: {len(annotation_doc['elements'])}")

    label_json = tempfile.NamedTemporaryFile(suffix=".json").name
    with open(label_json, "w") as fp:
        json.dump(annotation_doc, fp)
    return label_json, len(annotation_doc["elements"])


def create_dsa_annotations_json(json_data, loglevel="INFO"):
    logger.setLevel(loglevel.upper())

    total_count = 0
    label_json = tempfile.NamedTemporaryFile(suffix=".json").name
    with open(label_json, "w") as fp:
        fp.write("{\n")
        fp.write(' "name": "{}",\n'.format(json_data["name"]))
        fp.write(' "description": "{}",\n'.format(json_data["description"]))
        fp.write(' "elements": [\n')

        for tid, res in enumerate(json_data["annotations"]):
            logger.debug(f"Adding annotations for tile: {tid}")
            if not res:
                continue

            annotation = res.get("annotation")
            if not annotation:
                continue

            color_map = annotation.get("labels")
            elements = annotation.get("elements", [])
            for element in elements:
                label = element["label"]
                color = to_rgb(color_map.get(label))

                logger.info(f"Adding Contours for label: {label}; color: {color}; color_map: {color_map}")

                contours = element["contours"]
                for contour in contours:
                    points = []
                    for point in contour:
                        points.append([point[0], point[1], 0])

                    if total_count > 0:
                        fp.write(",\n")

                    annotation_style = {
                        "group": label,
                        "type": "polyline",
                        "lineColor": color,
                        "lineWidth": 2.0,
                        "closed": True,
                        "points": points,
                        "label": {"value": label},
                    }
                    fp.write(f"  {json.dumps(annotation_style)}")
                    total_count += 1

        fp.write(" ]\n")  # close elements
        fp.write("}")  # end of root

    logger.info(f"Total Elements: {total_count}")
    return label_json, total_count


def create_asap_annotations_xml(json_data, loglevel="INFO"):
    logger.setLevel(loglevel.upper())

    total_count = 0
    label_xml = tempfile.NamedTemporaryFile(suffix=".xml").name

    name = json_data["name"]
    description = json_data["description"]
    model = json_data["model"]
    location = json_data["location"]
    size = json_data["size"]

    with open(label_xml, "w") as fp:
        fp.write('<?xml version="1.0"?>\n')
        fp.write("<ASAP_Annotations>\n")
        fp.write(
            '  <Annotations Name="{}" Description="{}" Model="{}" X="{}" Y="{}" W="{}" H="{}">\n'.format(
                name, description, model, location[0], location[1], size[0], size[1]
            )
        )

        labels = {}
        for tid, res in enumerate(json_data["annotations"]):
            logger.debug(f"Adding annotations for tile: {tid}")
            if not res:
                continue

            annotation = res.get("annotation")
            if not annotation:
                continue

            logger.info(f"Annotation keys: {annotation.keys()}")
            color_map = annotation.get("labels")
            elements = annotation.get("elements", [])
            for element in elements:
                label = element["label"]
                color = to_hex(color_map.get(label))

                logger.info(f"Adding Contours for label: {label}; color: {color}; color_map: {color_map}")
                labels[label] = color

                contours = element["contours"]
                for contour in contours:
                    fp.write(f'    <Annotation Name="{label}" Type="Polygon" PartOfGroup="{label}" Color="{color}">\n')
                    fp.write("      <Coordinates>\n")
                    for pcount, point in enumerate(contour):
                        fp.write(f'        <Coordinate Order="{pcount}" X="{point[0]}" Y="{point[1]}" />\n')
                    fp.write("      </Coordinates>\n")
                    fp.write("    </Annotation>\n")
                    total_count += 1
        fp.write("  </Annotations>\n")

        fp.write("  <AnnotationGroups>\n")
        for label, color in labels.items():
            fp.write(f'    <Group Name="{label}" PartOfGroup="None" Color="{color}">\n')
            fp.write("      <Attributes />\n")
            fp.write("    </Group>\n")
        fp.write("  </AnnotationGroups>\n")
        fp.write("</ASAP_Annotations>\n")

    logger.info(f"Total Annotations: {total_count}")
    return label_xml, total_count
