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
    image_np = np.moveaxis(image_np, 0, 1)

    pos_points = data.get("pos_points")
    neg_points = data.get("neg_points")

    json_data = context.user_data.model_handler.infer(
        request={
            "model": context.user_data.model,
            "image": image_np,
            "foreground": np.flip(np.array(pos_points, int), 1).tolist() if pos_points else pos_points,
            "background": np.flip(np.array(neg_points, int), 1).tolist() if neg_points else neg_points,
            "output": "json",
        }
    )

    results = []
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
                points = np.flip(np.array(contour, int), axis=None)

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
