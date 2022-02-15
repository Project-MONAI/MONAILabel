import logging
import tempfile

logger = logging.getLogger(__name__)


def create_annotations_xml(json_data):
    count = 0
    label_xml = tempfile.NamedTemporaryFile(suffix=".xml").name
    with open(label_xml, "w") as fp:
        fp.write('<?xml version="1.0"?>\n')
        fp.write("<ASAP_Annotations>\n")
        fp.write("  <Annotations>\n")
        for k, v in json_data.items():
            contours = v["contours"]
            for count, contour in enumerate(contours):
                fp.write(
                    '    <Annotation Name="Tumor" Type="Polygon" PartOfGroup="Tumor" Color="#F4FA58">\n'.format(
                        k, count
                    )
                )
                fp.write("      <Coordinates>\n")
                for pcount, point in enumerate(contour):
                    fp.write('        <Coordinate Order="{}" X="{}" Y="{}" />\n'.format(pcount, point[0], point[1]))
                fp.write("      </Coordinates>\n")
                count += 1
                fp.write("    </Annotation>\n")
        fp.write("  </Annotations>\n")
        fp.write("  <AnnotationGroups>\n")
        fp.write('    <Group Name="Tumor" PartOfGroup="None" Color="#00ff00">\n')
        fp.write("      <Attributes />\n")
        fp.write("    </Group>\n")
        fp.write("  </AnnotationGroups>\n")
        fp.write("</ASAP_Annotations>\n")

    logger.error(f"Total Polygons: {count}")
    return label_xml
