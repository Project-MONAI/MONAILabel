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
import copy
import logging
import os
import pathlib
import shutil
from time import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from monai.transforms import LoadImaged
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.util import img_as_ubyte
from tqdm import tqdm

from monailabel.config import settings
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.utils.transform import run_transforms
from monailabel.transform.writer import Writer
from monailabel.utils.others.generic import device_list, download_file, get_basename_no_ext, name_to_device, strtobool

logger = logging.getLogger(__name__)


class Sam2InferTask(InferTask):
    def __init__(
        self,
        model_dir,
        type=InferType.DEEPGROW,
        dimension=2,
        labels=None,
        additional_info=None,
        image_loader=LoadImaged(keys="image"),
        post_trans=None,
        writer=Writer(ref_image="image"),
        config=None,
    ):
        super().__init__(
            type=type,
            dimension=dimension,
            labels=labels,
            description="SAM2 (Segment Anything Model)",
            config={"device": device_list(), "reset_state": False},
        )
        self.additional_info = additional_info
        self.image_loader = image_loader
        self.post_trans = post_trans
        self.writer = writer
        if config:
            self._config.update(config)

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

    def info(self) -> Dict[str, Any]:
        d = super().info()
        if self.additional_info:
            d.update(self.additional_info)
        return d

    def is_valid(self) -> bool:
        return True

    def run2d(self, image_tensor, request, debug=False):
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

        slices = {p[2] for p in request["foreground"] if len(p) > 2}
        slices.update({p[2] for p in request["background"] if len(p) > 2})
        slices = list(slices)
        slice_idx = slices[0] if len(slices) else -1
        logger.info(f"Slices: {slices}; Slice Index: {slice_idx}")

        if slice_idx < 0:
            slice_np = image_tensor.cpu().numpy()
            slice_rgb_np = slice_np.astype(np.uint8) if np.max(slice_np) > 1 else img_as_ubyte(slice_np)
        else:
            slice_np = image_tensor[:, :, slice_idx].cpu().numpy()
            slice_rgb_np = np.array(Image.fromarray(slice_np).convert("RGB"))

        logger.info(f"Slice Index:{slice_idx}; (Image) Slice Shape: {slice_np.shape}")
        print(f"Slice: Type: {slice_np.dtype}; Max: {np.max(slice_np)}")
        print(f"Slice RGB: Type: {slice_rgb_np.dtype}; Max: {np.max(slice_rgb_np)}")
        if debug:
            shutil.copy(image_tensor.meta["filename_or_obj"], "image.jpg")
            Image.fromarray(slice_rgb_np).save("slice.jpg")

        predictor.reset_predictor()
        predictor.set_image(slice_rgb_np)

        location = request.get("location", (0, 0))
        tx, ty = location[0], location[1]
        fp = [[p[1] - ty, p[0] - tx] for p in request["foreground"]]
        bp = [[p[1] - ty, p[0] - tx] for p in request["background"]]

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
        if self.post_trans is None:
            if slice_idx < 0:
                pred = masks[0]
            else:
                pred = np.zeros(tuple(image_tensor.shape))
                pred[:, :, slice_idx] = masks[0]

            data = copy.copy(request)
            data.update({"image_path": request["image"], "pred": pred, "image": image_tensor})
        else:
            data = copy.copy(request)
            data.update({"image_path": request["image"], "pred": masks[0], "image": image_tensor})
            data = run_transforms(data, self.post_trans, log_prefix="POST", use_compose=False)

        if debug:
            # pylab.imsave("mask.jpg", masks[0], format="jpg", cmap="Greys_r")
            Image.fromarray(masks[0] > 0).save("mask.jpg")

        return self.writer(data)

    def run_3d(self, image_tensor, set_image_state, request):
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
        data = copy.copy(request)
        data.update({"image_path": request["image"], "pred": pred, "image": image_tensor})
        return writer(data)

    def __call__(self, request, debug=False) -> Tuple[Union[str, None], Dict]:
        start_ts = time()

        logger.info(f"Infer Request: {request}")
        image_path = request["image"]
        image_tensor = self.image_cache.get(image_path)
        set_image_state = False
        cache_image = request.get("cache_image", True)

        if "foreground" not in request:
            request["foreground"] = []
        if "background" not in request:
            request["background"] = []

        if request.get("flip_points", False):
            request["foreground"] = [[p[1], p[0]] + p[2:] for p in request["foreground"]]
            request["background"] = [[p[1], p[0]] + p[2:] for p in request["background"]]

        if not cache_image or image_tensor is None:
            # TODO:: Fix this to cache more than one image session
            self.image_cache.clear()
            image_tensor = self.image_loader(request)["image"]
            logger.info(f"Image Meta: {image_tensor.meta}")
            self.image_cache[image_path] = image_tensor
            set_image_state = True

            video_dir = os.path.join(self.cache_path, get_basename_no_ext(image_path))
            if self.dimension == 3 and not os.path.isdir(video_dir):
                os.makedirs(video_dir, exist_ok=True)
                for slice_idx in tqdm(range(image_tensor.shape[-1])):
                    slice_np = image_tensor[:, :, slice_idx].numpy()
                    slice_file = os.path.join(video_dir, f"{str(slice_idx).zfill(5)}.jpg")

                    # pylab.imsave(slice_file, slice_np, format="jpg", cmap="Greys_r")
                    Image.fromarray(slice_np).convert("RGB").save(slice_file)
                logger.info(f"Image (Flattened): {image_tensor.shape[-1]} slices")

        logger.info(f"Image Shape: {image_tensor.shape}; cached: {cache_image}")
        if self.dimension == 2:
            mask_file, result_json = self.run2d(image_tensor, request, debug)
        else:
            mask_file, result_json = self.run_3d(image_tensor, set_image_state, request)

        logger.info(f"Mask File: {mask_file}; Latency: {round(time() - start_ts, 4)} sec")
        result_json["latencies"] = {
            "pre": 0,
            "infer": 0,
            "invert": 0,
            "post": 0,
            "write": 0,
            "total": round(time() - start_ts, 2),
            "transform": None,
        }
        return mask_file, result_json


"""
def main():
    import shutil

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    app_name = "pathology"
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sample-apps", app_name))
    model_dir = os.path.join(app_dir, "model")
    logger.info(f"Model Dir: {model_dir}")
    if app_name == "pathology":
        from lib.transforms import LoadImagePatchd

        from monailabel.transform.post import FindContoursd
        from monailabel.transform.writer import PolygonWriter

        task = Sam2InferTask(
            model_dir=model_dir,
            dimension=2,
            additional_info={"nuclick": True, "pathology": True},
            image_loader=LoadImagePatchd(keys="image", padding=False),
            post_trans=[FindContoursd(keys="pred")],
            writer=PolygonWriter(),
        )
        request = {
            "device": "cuda:1",
            "reset_state": False,
            "model": "sam2",
            "image": "/home/sachi/Datasets/wsi/JP2K-33003-1.svs",
            "output": "asap",
            "level": 0,
            "location": (2183, 4873),
            "size": (128, 128),
            "tile_size": [128, 128],
            "min_poly_area": 30,
            "foreground": [[2247, 4937]],
            "background": [],
            "max_workers": 1,
            "id": 0,
            "logging": "INFO",
            "result_write_to_file": False,
            "description": "SAM2 (Segment Anything Model)",
            "save_label": False,
        }
    else:
        task = Sam2InferTask(model_dir)
        request = {
            "image": "/home/sachi/Datasets/SAM2/spleen_16.nii.gz",
            "foreground": [[129, 199, 45], [129, 199, 47], [100, 200, 41]],
            "background": [[199, 129, 45], [399, 129, 45]],
        }

    result = task(request, debug=True)
    shutil.move(result[0], "/home/sachi/" + "mask.xml" if app_name == "pathology" else "mask.nii.gz")


if __name__ == "__main__":
    main()
"""
