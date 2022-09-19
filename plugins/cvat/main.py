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

import base64
import io
import json
import logging
import os
from distutils.util import strtobool

import numpy as np
from PIL import Image

from monailabel.interfaces.utils.app import app_instance

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init_context(context):
    context.logger.info("Init context...  0%")

    app_dir = os.environ.get("MONAI_LABEL_APP_DIR", "/opt/conda/monailabel/sample-apps/pathology")
    studies = os.environ.get("MONAI_LABEL_STUDIES", "/opt/monailabel/studies")
    model = os.environ.get("MONAI_LABEL_MODELS", "segmentation_nuclei")
    pretrained_path = os.environ.get(
        "MONAI_PRETRAINED_PATH", "https://github.com/Project-MONAI/MONAILabel/releases/download/data"
    )
    conf = {"preload": "true", "models": model, "pretrained_path": pretrained_path}

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    app_dir = app_dir if os.path.exists(app_dir) else os.path.join(root_dir, "sample-apps", "pathology")
    studies = studies if os.path.exists(os.path.dirname(studies)) else os.path.join(root_dir, "studies")

    app = app_instance(app_dir, studies, conf)

    context.user_data.model = model
    context.user_data.model_handler = app
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info(f"Run model: {context.user_data.model}")
    data = event.body

    image = Image.open(io.BytesIO(base64.b64decode(data["image"])))
    image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)

    flip_image = strtobool(os.environ.get("MONAI_LABEL_FLIP_INPUT_IMAGE", "true"))
    flip_points = strtobool(os.environ.get("MONAI_LABEL_FLIP_INPUT_POINTS", "true"))
    flip_output = strtobool(os.environ.get("MONAI_LABEL_FLIP_OUTPUT_POINTS", "true"))

    if flip_image:
        image_np = np.moveaxis(image_np, 0, 1)

    pos_points = data.get("pos_points")
    neg_points = data.get("neg_points")
    if flip_points:
        foreground = np.flip(np.array(pos_points, int), 1).tolist() if pos_points else pos_points
        background = np.flip(np.array(neg_points, int), 1).tolist() if neg_points else neg_points
    else:
        foreground = np.array(pos_points, int).tolist() if pos_points else pos_points
        background = np.array(neg_points, int).tolist() if neg_points else neg_points

    context.logger.info(f"Image: {image_np.shape}; Foreground: {foreground}; Background: {background}")

    json_data = context.user_data.model_handler.infer(
        request={
            "model": context.user_data.model,
            "image": image_np,
            "foreground": foreground,
            "background": background,
            "output": "json",
        }
    )

    results = []
    prediction = json_data["params"].get("prediction")
    if prediction:
        context.logger.info(f"(Classification) Prediction: {prediction}")

        # CVAT Limitation:: tag is not yet supported https://github.com/opencv/cvat/issues/4212
        # CVAT Limitation:: select highest score and create bbox to represent as tag
        e = None
        for element in prediction:
            if element["score"] > 0:
                e = element if e is None or element["score"] > e["score"] else e
                context.logger.info(f"New Max Element: {e}")

        context.logger.info(f"Final Element with Max Score: {e}")
        if e:
            results.append(
                {
                    "label": e["label"],
                    "confidence": e["score"],
                    "type": "rectangle",
                    "points": [0, 0, image_np.shape[0] - 1, image_np.shape[1] - 1],
                }
            )
        context.logger.info(f"(Classification) Results: {results}")
    else:
        interactor = strtobool(os.environ.get("INTERACTOR_MODEL", "false"))
        annotations = json_data["params"].get("annotations")
        for a in annotations:
            annotation = a.get("annotation", {})
            if not annotation:
                continue

            elements = annotation.get("elements", [])
            for element in elements:
                label = element["label"]
                contours = element["contours"]
                for contour in contours:
                    points = np.array(contour, int)
                    if flip_output:
                        points = np.flip(points, axis=None)

                    # CVAT limitation:: only one polygon result for interactor
                    if interactor and contour:
                        return context.Response(
                            body=json.dumps(points.tolist()),
                            headers={},
                            content_type="application/json",
                            status_code=200,
                        )

                    results.append(
                        {
                            "label": label,
                            "points": points.flatten().tolist(),
                            "type": "polygon",
                        }
                    )

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )


"""
if __name__ == "__main__":
    import logging
    from argparse import Namespace

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    context = {
        "logger": logging.getLogger(__name__),
        "user_data": Namespace(**{"model": None, "model_handler": None}),
    }
    context = Namespace(**context)

    with open("test.jpg", "rb") as fp:
        image = base64.b64encode(fp.read())

    event = {
        "body": {
            "image": image,
        }
    }
    event = Namespace(**event)

    init_context(context)
    response = handler(context, event)
    print(response)
"""
