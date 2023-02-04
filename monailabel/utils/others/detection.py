# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import tempfile

logger = logging.getLogger(__name__)


def create_slicer_detection_json(json_data, loglevel="INFO"):
    logger.setLevel(loglevel.upper())

    total_count = 0
    label_json = tempfile.NamedTemporaryFile(suffix=".json").name

    with open(label_json, "w") as fp:
        fp.write("{\n")
        fp.write(
            ' "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",\n'
        )

        fp.write(' "markups": [\n')

        for idx, item in enumerate(json_data["box"]):
            center = item[0:3]
            size = item[3:]
            orientation = [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0]
            label = json_data["label"][idx]

            if total_count > 0:
                fp.write(",\n")

            control_points = {
                "id": "1",
                "label": "ROI",
                "description": "",
                "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                "position": center,
                "orientation": orientation,
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }
            measurements = {"name": "volume", "enabled": False, "units": "cm3", "printFormat": "%-#4.4g%s"}
            detection_node = {
                "name": json_data["image"].split("/")[-1],
                "type": "ROI",
                "coordinateSystem": "LPS",  # use LPS coordinate system by default, which is defined by the bundle
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 1,
                "roiType": "Box",
                "center": center,
                "orientation": orientation,
                "size": size,
                "insideOut": False,
                "label": {"value": label},
                "controlPoints": [control_points],
                "measurements": [measurements],
            }
            fp.write(f"  {json.dumps(detection_node)}")
            total_count += 1

        fp.write("]\n")  # close elements
        fp.write("}")  # end of root

    logger.info(f"Total Elements: {total_count}")
    return label_json, total_count
