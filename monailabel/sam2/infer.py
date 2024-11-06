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
import tempfile
from datetime import timedelta
from time import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import pylab
import schedule
import torch
from monai.transforms import KeepLargestConnectedComponent, LoadImaged
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.util import img_as_ubyte
from timeloop import Timeloop
from tqdm import tqdm

from monailabel.config import settings
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.utils.transform import run_transforms
from monailabel.transform.writer import Writer
from monailabel.utils.others.generic import (
    device_list,
    download_file,
    get_basename_no_ext,
    md5_digest,
    name_to_device,
    remove_file,
    strtobool,
)

logger = logging.getLogger(__name__)


class ImageCache:
    def __init__(self):
        cache_path = settings.MONAI_LABEL_DATASTORE_CACHE_PATH
        self.cache_path = (
            os.path.join(cache_path, "sam2")
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "sam2")
        )
        self.cached_dirs = {}
        self.cache_expiry_sec = 10 * 60

        remove_file(self.cache_path)
        os.makedirs(self.cache_path, exist_ok=True)
        logger.info(f"Image Cache Initialized: {self.cache_path}")

    def cleanup(self):
        ts = time()
        expired = {k: v for k, v in self.cached_dirs.items() if v < ts}
        for k, v in expired.items():
            self.cached_dirs.pop(k)
            logger.info(f"Remove Expired Image: {k}; ExpiryTs: {v}; CurrentTs: {ts}")
            remove_file(k)

    def monitor(self):
        self.cleanup()
        time_loop = Timeloop()
        schedule.every(1).minutes.do(self.cleanup)

        @time_loop.job(interval=timedelta(seconds=60))
        def run_scheduler():
            schedule.run_pending()

        time_loop.start(block=False)


image_cache = ImageCache()
image_cache.monitor()


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
            config={"device": device_list(), "reset_state": False, "largest_cc": False, "pylab": False},
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

        slice_idx = request.get("slice")
        if slice_idx is None or slice_idx < 0:
            slices = {p[2] for p in request["foreground"] if len(p) > 2}
            slices.update({p[2] for p in request["background"] if len(p) > 2})
            slices = list(slices)
            slice_idx = slices[0] if len(slices) else -1
        else:
            slices = {slice_idx}

        if slice_idx < 0 and len(request["roi"]) == 6:
            slice_idx = round(request["roi"][4] + (request["roi"][5] - request["roi"][4]) // 2)
            slices = {slice_idx}
        logger.info(f"Slices: {slices}; Slice Index: {slice_idx}")

        if slice_idx < 0:
            slice_np = image_tensor.cpu().numpy()
            slice_rgb_np = slice_np.astype(np.uint8) if np.max(slice_np) > 1 else img_as_ubyte(slice_np)
        else:
            slice_np = image_tensor[:, :, slice_idx].cpu().numpy()

            if strtobool(request.get("pylab")):
                slice_rgb_file = tempfile.NamedTemporaryFile(suffix=".jpg").name
                pylab.imsave(slice_rgb_file, slice_np, format="jpg", cmap="Greys_r")
                slice_rgb_np = np.array(Image.open(slice_rgb_file))
                remove_file(slice_rgb_file)
            else:
                slice_rgb_np = np.array(Image.fromarray(slice_np).convert("RGB"))

        logger.info(f"Slice Index:{slice_idx}; (Image) Slice Shape: {slice_np.shape}")
        if debug:
            logger.info(f"Slice {slice_np.shape} Type: {slice_np.dtype}; Max: {np.max(slice_np)}")
            logger.info(f"Slice RGB {slice_rgb_np.shape} Type: {slice_rgb_np.dtype}; Max: {np.max(slice_rgb_np)}")
            if slice_idx < 0 and image_tensor.meta.get("filename_or_obj"):
                shutil.copy(image_tensor.meta["filename_or_obj"], "image.jpg")
            else:
                pylab.imsave("image.jpg", slice_np, format="jpg", cmap="Greys_r")
            Image.fromarray(slice_rgb_np).save("slice.jpg")

        predictor.reset_predictor()
        predictor.set_image(slice_rgb_np)

        location = request.get("location", (0, 0))
        tx, ty = location[0], location[1]
        fp = [[p[0] - tx, p[1] - ty] for p in request["foreground"]]
        bp = [[p[0] - tx, p[1] - ty] for p in request["background"]]
        roi = request.get("roi")
        roi = [roi[0] - tx, roi[1] - ty, roi[2] - tx, roi[3] - ty] if roi else None

        if debug:
            slice_rgb_np_p = np.copy(slice_rgb_np)
            if roi:
                slice_rgb_np_p[roi[0] : roi[2], roi[1] : roi[3], 2] = 255
            for k, ps in {1: fp, 0: bp}.items():
                for p in ps:
                    slice_rgb_np_p[p[0] - 2 : p[0] + 2, p[1] - 2 : p[1] + 2, k] = 255
            Image.fromarray(slice_rgb_np_p).save("slice_p.jpg")

        point_coords = fp + bp
        point_coords = [[p[1], p[0]] for p in point_coords]  # Flip x,y => y,x
        box = [roi[1], roi[0], roi[3], roi[2]] if roi else None

        point_labels = [1] * len(fp) + [0] * len(bp)
        logger.info(f"Coords: {point_coords}; Labels: {point_labels}; Box: {box}")

        masks, scores, _ = predictor.predict(
            point_coords=np.array(point_coords) if point_coords else None,
            point_labels=np.array(point_labels) if point_labels else None,
            multimask_output=False,
            box=np.array(box) if box else None,
        )
        # sorted_ind = np.argsort(scores)[::-1]
        # masks = masks[sorted_ind]
        # scores = scores[sorted_ind]
        if strtobool(request.get("largest_cc", False)):
            masks = KeepLargestConnectedComponent()(masks).cpu().numpy()

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

    def run_3d(self, image_tensor, set_image_state, request, debug=False):
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

        image_path = request["image"]
        video_dir = os.path.join(
            image_cache.cache_path, get_basename_no_ext(image_path) if debug else md5_digest(image_path)
        )
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            for slice_idx in tqdm(range(image_tensor.shape[-1])):
                slice_np = image_tensor[:, :, slice_idx].numpy()
                slice_file = os.path.join(video_dir, f"{str(slice_idx).zfill(5)}.jpg")

                if strtobool(request.get("pylab")):
                    pylab.imsave(slice_file, slice_np, format="jpg", cmap="Greys_r")
                else:
                    Image.fromarray(slice_np).convert("RGB").save(slice_file)
            logger.info(f"Image (Flattened): {image_tensor.shape[-1]} slices; {video_dir}")

        # Set Expiry Time
        image_cache.cached_dirs[video_dir] = time() + image_cache.cache_expiry_sec

        if reset_state or set_image_state:
            if self.inference_state:
                predictor.reset_state(self.inference_state)
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
                    kps[sid].append([p[0], p[1]])
                else:
                    kps[sid] = [[p[0], p[1]]]

        box = None
        roi = request.get("roi")
        if roi:
            box = [roi[1], roi[0], roi[3], roi[2]]
            sids.update([i for i in range(roi[4], roi[5])])

        pred = np.zeros(tuple(image_tensor.shape))
        for sid in sorted(sids):
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])

            point_coords = fp + bp
            point_coords = [[p[1], p[0]] for p in point_coords]  # Flip x,y => y,x
            point_labels = [1] * len(fp) + [0] * len(bp)
            # logger.info(f"{sid} - Coords: {point_coords}; Labels: {point_labels}; Box: {box}")

            o_frame_ids, o_obj_ids, o_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=sid,
                obj_id=1,
                points=np.array(point_coords) if point_coords else None,
                labels=np.array(point_labels) if point_labels else None,
                box=np.array(box) if box else None,
            )

            # logger.info(f"{sid} - mask_logits: {o_mask_logits.shape}; frame_ids: {o_frame_ids}; obj_ids: {o_obj_ids}")
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
        if "roi" not in request:
            request["roi"] = []

        if not cache_image or image_tensor is None:
            # TODO:: Fix this to cache more than one image session
            self.image_cache.clear()
            image_tensor = self.image_loader(request)["image"]
            if debug:
                logger.info(f"Image Meta: {image_tensor.meta}")
            self.image_cache[image_path] = image_tensor
            set_image_state = True

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
            # "roi": [2220, 4900, 2320, 5000],
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
            "image": "/home/sachi/Datasets/SAM2/image.nii.gz",
            "foreground": [[71, 175, 105]],  # [199, 129, 47], [200, 100, 41]],
            # "background": [[286, 175, 105]],
            "roi": [44, 110, 113, 239, 72, 178],
            "largest_cc": True,
        }

    result = task(request, debug=True)
    if app_name == "pathology":
        print(result)
    else:
        shutil.move(result[0], "mask.nii.gz")


if __name__ == "__main__":
    main()
"""
