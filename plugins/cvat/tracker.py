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

from PIL import Image

from monailabel.client import MONAILabelClient

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init_context(context):
    context.logger.info("Init context...  0%")
    server = os.environ.get("MONAI_LABEL_SERVER", "http://0.0.0.0:8000")
    model = os.environ.get("MONAI_LABEL_MODEL", "sam2")
    client = MONAILabelClient(server)

    info = client.info()
    model_info = info["models"][model] if info and info["models"] else None
    context.logger.info(f"Monai Label Info: {model_info}")
    assert model_info

    context.user_data.model = model
    context.user_data.model_handler = client
    context.logger.info("Init context...100%")


def handler(context, event):
    model: str = context.user_data.model
    client: MONAILabelClient = context.user_data.model_handler
    context.logger.info(f"Run model: {model}")

    data = event.body
    image = Image.open(io.BytesIO(base64.b64decode(data["image"])))
    context.logger.info(f"Image: {image.size}")
    context.logger.info(f"Event Data Keys: {data.keys()}")

    image_file = tempfile.NamedTemporaryFile(suffix=".jpg").name
    image.save(image_file)

    shapes = data.get("shapes")
    context.logger.info(f"Shapes: {shapes}")

    states = data.get("states")
    context.logger.info(f"States: {states}")

    # server = os.environ.get("MONAI_LABEL_SERVER", "http://0.0.0.0:8000")
    # model = os.environ.get("MONAI_LABEL_MODEL", "sam2T")

    # client = MONAILabelClient(server)
    # params = {"output": "mask"}

    # output_mask, output_json = client.infer(model=model, image_id="", file=image_file, params=params)
    # if isinstance(output_json, str) or isinstance(output_json, bytes):
    #     output_json = json.loads(output_json)
    # context.logger.info(f"Mask: {output_mask}; Output JSON: {output_json}")

    # mask_np = np.array(Image.open(output_mask)).astype(np.uint8)
    # context.logger.info(f"Mask: {mask_np.shape}")
    # os.remove(output_mask)
    os.remove(image_file)

    results = {"shapes": [], "states": []}
    for i, shape in enumerate(shapes):
        # shape, state = context.user_data.model.infer(image, shape, states[i] if i < len(states) else None)
        # results['shapes'].append(shape)
        # results['states'].append(state)
        print(i, shape)

    # context.logger.info(f"Image: {image.size}; Mask: {mask_np.shape}; JSON: {output_json}")
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

    event = {"body": {"image": image}}
    event = Namespace(**event)

    init_context(context)
    response = handler(context, event)
    print(response)
