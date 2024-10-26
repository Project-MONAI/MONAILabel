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
import tempfile

import numpy as np
from PIL import Image

from monailabel.client import MONAILabelClient
from monailabel.utils.others.generic import strtobool

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init_context(context):
    context.logger.info("Init context...  0%")

    server = os.environ.get("MONAI_LABEL_SERVER", "http://0.0.0.0:8000")
    model = os.environ.get("MONAI_LABEL_MODEL", "deepedit")
    client = MONAILabelClient(server)

    info = client.info()
    model_info = info["models"][model] if info and info["models"] else None
    context.logger.info(f"Monai Label Info: {model_info}")
    assert model_info

    context.user_data.model = model
    context.user_data.model_handler = client
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info(f"Run model: {context.user_data.model}")
    data = event.body

    image = Image.open(io.BytesIO(base64.b64decode(data["image"])))
    foreground = data.get("pos_points")
    background = data.get("neg_points")
    context.logger.info(f"Image: {image.size}; Foreground: {foreground}; Background: {background}")

    image_file = tempfile.NamedTemporaryFile(suffix=".jpg").name
    image.save(image_file)

    server = os.environ.get("MONAI_LABEL_SERVER", "http://0.0.0.0:8000")
    model = os.environ.get("MONAI_LABEL_MODEL", "inbody")
    output = os.environ.get("MONAI_LABEL_OUTPUT_TYPE", "mask")

    client = MONAILabelClient(server)
    params = {
        "output": output,
        "foreground": np.asarray(foreground, dtype=int).tolist() if foreground else [],
        "background": np.asarray(background, dtype=int).tolist() if background else [],
    }

    output_mask, output_json = client.infer(model=model, image_id="", file=image_file, params=params)
    if isinstance(output_json, str) or isinstance(output_json, bytes):
        output_json = json.loads(output_json)
    # context.logger.info(f"Mask: {output_mask}; Output JSON: {output_json}")

    interactor = strtobool(os.environ.get("INTERACTOR_MODEL", "false"))
    if interactor:
        mask_np = np.array(Image.open(output_mask)).astype(np.uint8)
        context.logger.info(f"Mask: {mask_np.shape}")
        os.remove(output_mask)

        resp = {"mask": mask_np.tolist()}
        context.logger.info(f"Image: {image.size}; Mask: {mask_np.shape}; JSON: {output_json}")
        return context.Response(
            body=json.dumps(resp),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    results = []
    prediction = output_json.get("prediction")
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
                    "points": [0, 0, image.size[0] - 1, image.size[1] - 1],
                }
            )
        context.logger.info(f"(Classification) Results: {results}")
    else:
        annotations = output_json.get("annotations")
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

    with open("/home/sachi/Datasets/endo/frame001.jpg", "rb") as fp:
        image = base64.b64encode(fp.read())

    event = {
        "body": {
            "image": image,
            "pos_points": [[1209, 493]],
        }
    }
    event = Namespace(**event)

    init_context(context)
    response = handler(context, event)
    print(response)
