import logging
import tempfile

from shapely.geometry import Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def merge_polygons(polygons, min_poly_area=3000):
    logger.info(f"++++ Total Polygons: {len(polygons)}")
    ignored = 0
    try:
        multipolygon = unary_union([Polygon(p) for p in polygons])
    except ValueError:
        return polygons

    new_polygons = []
    for polygon in multipolygon.geoms:
        if polygon.area < min_poly_area:
            ignored += 1
            continue
        new_polygons.append(list(polygon.exterior.coords))

    logger.warning(f"Total Polygons: {len(polygons)}; Union Poly: {len(new_polygons)}; Ignored: {ignored}")
    return new_polygons


def create_annotations_xml(json_data, union=False, min_poly_area=3000):
    total_count = 0
    label_xml = tempfile.NamedTemporaryFile(suffix=".xml").name

    if union:
        all_polygons = []
        for v in json_data.values():
            all_polygons.extend(v["contours"])

        json_data = {"all": {"contours": merge_polygons(all_polygons, min_poly_area)}}

    with open(label_xml, "w") as fp:
        fp.write('<?xml version="1.0"?>\n')
        fp.write("<ASAP_Annotations>\n")
        fp.write("  <Annotations>\n")
        for v in json_data.values():
            contours = v["contours"]
            for contour in contours:
                fp.write('    <Annotation Name="Tumor" Type="Polygon" PartOfGroup="Tumor" Color="#F4FA58">\n')
                fp.write("      <Coordinates>\n")
                for pcount, point in enumerate(contour):
                    fp.write('        <Coordinate Order="{}" X="{}" Y="{}" />\n'.format(pcount, point[0], point[1]))
                fp.write("      </Coordinates>\n")
                fp.write("    </Annotation>\n")
                total_count += 1
        fp.write("  </Annotations>\n")
        fp.write("  <AnnotationGroups>\n")
        fp.write('    <Group Name="Tumor" PartOfGroup="None" Color="#00ff00">\n')
        fp.write("      <Attributes />\n")
        fp.write("    </Group>\n")
        fp.write("  </AnnotationGroups>\n")
        fp.write("</ASAP_Annotations>\n")

    logger.error(f"Total Polygons: {total_count}")
    return label_xml
