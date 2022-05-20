import base64
import io
import json
import os

import numpy as np
from PIL import Image

from monailabel.interfaces.utils.app import app_instance


def init_context(context):
    context.logger.info("Init context...  0%")

    app_dir = os.environ.get("MONAI_LABEL_APP_DIR", "/opt/conda/monailabel/sample-apps/pathology")
    studies = os.environ.get("MONAI_LABEL_STUDIES", "/opt/monailabel/studies")
    model = os.environ.get("MONAI_LABEL_MODELS", "segmentation_nuclei")
    conf = {"preload": "true", "models": model}

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    app_dir = app_dir if os.path.exists(app_dir) else os.path.join(root_dir, "sample-apps", "pathology")
    studies = studies if os.path.exists(os.path.dirname(studies)) else os.path.join(root_dir, "studies")

    app = app_instance(app_dir, studies, conf)

    context.user_data.model = model
    context.user_data.model_handler = app
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run Segmentation Nuclei model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    json_data = context.user_data.model_handler.infer(
        request={
            "model": context.user_data.model,
            "image": image,
            "output": "json",
            "result_write_to_file": False,
        }
    )
    # print(json_data)

    results = []
    annotation = json_data["params"].get("annotation")
    if annotation:
        elements = annotation.get("elements", [])
        for element in elements:
            label = element["label"]
            contours = element["contours"]
            for contour in contours:
                results.append({
                    "label": label,
                    "points": contour,
                    "type": "polygon",
                })

    # return json.dumps(results)
    return context.Response(body=json.dumps(results), headers={}, content_type='application/json', status_code=200)

#
# if __name__ == "__main__":
#     import logging
#     from argparse import Namespace
#
#     logging.basicConfig(
#         level=logging.INFO,
#         format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#
#     context = {
#         "logger": logging.getLogger(__name__),
#         "user_data": Namespace(**{"model": None, "model_handler": None}),
#     }
#     context = Namespace(**context)
#
#     with open("test.png", "rb") as fp:
#         image = base64.b64encode(fp.read())
#
#     event = {
#         "body": {
#             "image": image,
#         }
#     }
#     event = Namespace(**event)
#
#     init_context(context)
#     response = handler(context, event)
#     print(response)
