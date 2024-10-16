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

import logging
import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from monai.transforms import LoadImage
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.transform.writer import Writer
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)


class Sam2InferTask(InferTask):
    def __init__(self, model_dir, dimension=2):
        super().__init__(
            type=InferType.DEEPGROW,
            labels=None,
            dimension=dimension,
            description="SAM V2",
            config=None,
        )

        # Download PreTrained Model
        # https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description
        pt = "sam2.1_hiera_large.pt"
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{pt}"
        self.path = os.path.join(model_dir, f"pretrained_{pt}")
        download_file(url, self.path)

        self.config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"Using Device: {device}")

        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        logger.info(f"Using Config: {self.config_path}")
        sam2_model = build_sam2(self.config_path, self.path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.image_cache = {}

    def is_valid(self) -> bool:
        return True

    def __call__(self, request, debug=False) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        logger.info(f"Infer Request: {request}")
        image_path = request["image"]
        image_tensor = self.image_cache.get(image_path)
        if image_tensor is None:
            # TODO:: Fix this to cache more than one image session
            self.image_cache.clear()

            image_tensor = LoadImage()(image_path)
            self.image_cache[image_path] = image_tensor

        spatial_shape = image_tensor.shape
        logger.info(f"Image Shape: {spatial_shape}")

        slice_idx = request["foreground"][0][2]
        logger.info(f"Slice Index: {slice_idx}")

        slice_np = image_tensor[:, :, slice_idx].numpy()
        logger.info(f"Image Slice Shape: {slice_np.shape}")
        slice_img = Image.fromarray(slice_np).convert("RGB")

        if debug:
            slice_img.save("slice.jpg")

        slice_rgb_np = np.array(slice_img)
        self.predictor.set_image(slice_rgb_np)

        fp = [[p[1], p[0]] for p in request["foreground"]]
        bp = [[p[1], p[0]] for p in request["background"]]

        if debug:
            slice_rgb_np_p = np.copy(slice_rgb_np)
            for k, ps in {1: fp, 0: bp}.items():
                for p in ps:
                    for i in (-2, -1, 0, 1, 2):
                        for j in (-2, -1, 0, 1, 2):
                            slice_rgb_np_p[p[1] + i, p[0] + j][k] = 255
            Image.fromarray(slice_rgb_np_p).save("slice_p.jpg")

        point_coords = np.array([fp + bp])
        point_labels = np.array([[1] * len(fp) + [0] * len(bp)])
        logger.info(f"Point Coords: {point_coords.tolist()}")
        logger.info(f"Point Labels: {point_labels.tolist()}")

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        logger.info(f"Masks Shape: {masks.shape}; Scores: {scores}")
        pred = np.zeros(tuple(spatial_shape))
        pred[:, :, slice_idx] = masks[0]

        if debug:
            # pylab.imsave("mask.jpg", masks[0], format="jpg", cmap="Greys_r")
            Image.fromarray(masks[0] > 0).save("mask.jpg")

        writer = Writer(ref_image="image")
        mask_file, result_json = writer({"image_path": request["image"], "pred": pred, "image": image_tensor})
        logger.info(f"Mask File: {mask_file}; Result JSON: {result_json}")

        return mask_file, result_json


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sample-apps", "radiology"))
    model_dir = os.path.join(app_dir, "model")
    logger.info(f"Model Dir: {model_dir}")
    task = Sam2InferTask(model_dir)

    sid = 45
    request = {
        "image": "/home/sachi/Datasets/SAM2/spleen_16.nii.gz",
        "foreground": [[129, 199, sid]],
        "background": [[199, 129, sid], [399, 129, sid]],
    }
    result = task(request, debug=True)
    print(result)


if __name__ == "__main__":
    main()
