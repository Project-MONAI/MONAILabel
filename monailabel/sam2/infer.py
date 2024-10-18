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
import pathlib
import shutil
from time import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from monai.transforms import LoadImage
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

from monailabel.config import settings
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.transform.writer import Writer
from monailabel.utils.others.generic import device_list, download_file, get_basename_no_ext, name_to_device, strtobool

logger = logging.getLogger(__name__)


class Sam2InferTask(InferTask):
    def __init__(self, model_dir, dimension=2):
        super().__init__(
            type=InferType.DEEPGROW,
            labels=None,
            dimension=dimension,
            description="SAM2 (Segment Anything Model)",
            config={"device": device_list(), "reset_state": False},
        )

        # Download PreTrained Model
        # https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description
        pt = "sam2.1_hiera_large.pt"
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{pt}"
        self.path = os.path.join(model_dir, f"pretrained_{pt}")
        download_file(url, self.path)

        self.config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictors = {}
        self.image_cache = {}
        self.inference_state = None

        cache_path = settings.MONAI_LABEL_DATASTORE_CACHE_PATH
        self.cache_path = (
            os.path.join(cache_path, "sam2")
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "sam2")
        )

    def is_valid(self) -> bool:
        return True

    def run2d(self, image_tensor, request, debug=False) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        device = name_to_device(request.get("device", "cuda"))
        predictor = self.predictors.get(device)
        if predictor is None:
            logger.info(f"Using Device: {device}")
            device_t = torch.device(device)
            if device_t.type == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            sam2_model = build_sam2(self.config_path, self.path, device=device)
            predictor = SAM2ImagePredictor(sam2_model)
            self.predictors[device] = predictor

        slice_idx = request["foreground"][0][2]
        logger.info(f"Slice Index: {slice_idx}")

        slice_np = image_tensor[:, :, slice_idx].numpy()
        logger.info(f"Image Slice Shape: {slice_np.shape}")
        slice_img = Image.fromarray(slice_np).convert("RGB")

        if debug:
            slice_img.save("slice.jpg")

        slice_rgb_np = np.array(slice_img)
        predictor.reset_predictor()
        predictor.set_image(slice_rgb_np)

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

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        logger.info(f"Masks Shape: {masks.shape}; Scores: {scores}")
        pred = np.zeros(tuple(image_tensor.shape))
        pred[:, :, slice_idx] = masks[0]

        if debug:
            # pylab.imsave("mask.jpg", masks[0], format="jpg", cmap="Greys_r")
            Image.fromarray(masks[0] > 0).save("mask.jpg")

        writer = Writer(ref_image="image")
        return writer({"image_path": request["image"], "pred": pred, "image": image_tensor})

    def run_3d(self, image_tensor, set_image_state, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        device = name_to_device(request.get("device", "cuda"))
        reset_state = strtobool(request.get("reset_state", "false"))
        predictor = self.predictors.get(device)
        if predictor is None:
            logger.info(f"Using Device: {device}")
            device_t = torch.device(device)
            if device_t.type == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            predictor = build_sam2_video_predictor(self.config_path, self.path, device=device)
            self.predictors[device] = predictor

        if reset_state or set_image_state:
            if self.inference_state:
                predictor.reset_state(self.inference_state)
            image_path = request["image"]
            video_dir = os.path.join(self.cache_path, get_basename_no_ext(image_path))
            self.inference_state = predictor.init_state(video_path=video_dir)

        logger.info(f"Image Shape: {image_tensor.shape}")
        fps: dict[int, Any] = {}
        bps: dict[int, Any] = {}
        sids = set()
        for key in {"foreground", "background"}:
            for p in request[key]:
                sid = p[2]
                sids.add(sid)
                kps = fps if key == "foreground" else bps
                if kps.get(sid):
                    kps[sid].append([p[1], p[0]])
                else:
                    kps[sid] = [[p[1], p[0]]]

        pred = np.zeros(tuple(image_tensor.shape))
        for sid in sids:
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])

            point_coords = np.array([fp + bp])
            point_labels = np.array([[1] * len(fp) + [0] * len(bp)])
            # logger.info(f"{sid} - Point Coords: {point_coords.tolist()}")
            # logger.info(f"{sid} - Point Labels: {point_labels.tolist()}")

            o_frame_ids, o_obj_ids, o_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=sid,
                obj_id=1,
                points=point_coords,
                labels=point_labels,
            )
            logger.info(f"{sid} - mask_logits: {o_mask_logits.shape}; frame_ids: {o_frame_ids}; obj_ids: {o_obj_ids}")
            pred[:, :, sid] = (o_mask_logits[0][0] > 0.0).cpu().numpy()

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            # logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()

        writer = Writer(ref_image="image")
        return writer({"image_path": request["image"], "pred": pred, "image": image_tensor})

    def __call__(self, request, debug=False) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        start_ts = time()

        logger.info(f"Infer Request: {request}")
        image_path = request["image"]
        image_tensor = self.image_cache.get(image_path)
        set_image_state = False
        if image_tensor is None:
            # TODO:: Fix this to cache more than one image session
            self.image_cache.clear()
            image_tensor = LoadImage()(image_path)
            self.image_cache[image_path] = image_tensor
            set_image_state = True

            video_dir = os.path.join(self.cache_path, get_basename_no_ext(image_path))
            if self.dimension == 3 and not os.path.isdir(video_dir):
                os.makedirs(video_dir, exist_ok=True)
                for slice_idx in tqdm(range(image_tensor.shape[-1])):
                    slice_np = image_tensor[:, :, slice_idx].numpy()
                    slice_img = Image.fromarray(slice_np).convert("RGB")
                    slice_img.save(os.path.join(video_dir, f"{str(slice_idx).zfill(5)}.jpg"))
                logger.info(f"Image (Flattened): {image_tensor.shape[-1]} slices")

        logger.info(f"Image Shape: {image_tensor.shape}")
        if self.dimension == 2:
            mask_file, result_json = self.run2d(image_tensor, request, debug)
        else:
            mask_file, result_json = self.run_3d(image_tensor, set_image_state, request)

        logger.info(f"Mask File: {mask_file}; Latency: {round(time() - start_ts, 4)} sec")
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

    request = {
        "image": "/home/sachi/Datasets/SAM2/spleen_16.nii.gz",
        "foreground": [[129, 199, 45], [129, 199, 47], [100, 200, 41]],
        "background": [[199, 129, 45], [399, 129, 45]],
    }
    result = task(request, debug=True)
    shutil.move(result[0], "/home/sachi/mask.nii.gz")


if __name__ == "__main__":
    main()
