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
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Callable

import torch

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.utils.others.generic import handle_torch_linalg_multithread, name_to_device, remove_file

logger = logging.getLogger(__name__)


class BatchInferImageType(str, Enum):
    IMAGES_ALL = "all"
    IMAGES_LABELED = "labeled"
    IMAGES_UNLABELED = "unlabeled"


class BatchInferTask:
    """
    Basic Batch Infer Task
    """

    def get_images(self, request, datastore: Datastore):
        """
        Override this method to get all eligible images for your task to run batch infer
        """
        images = request.get("images", BatchInferImageType.IMAGES_ALL)
        label_tag = request.get("label_tag")
        labels = request.get("labels")

        if isinstance(images, str):
            if images == BatchInferImageType.IMAGES_LABELED:
                return datastore.get_labeled_images(label_tag, labels)
            if images == BatchInferImageType.IMAGES_UNLABELED:
                return datastore.get_unlabeled_images(label_tag, labels)
            return datastore.list_images()
        return images

    def __call__(self, request, datastore: Datastore, infer: Callable):
        image_ids = sorted(self.get_images(request, datastore))
        max_batch_size = request.get("max_batch_size", 0)
        label_tag = request.get("label_tag", DefaultLabelTag.ORIGINAL)
        logger.info(
            f"Total number of images for batch inference: {len(image_ids)}; Max Batch Size: {max_batch_size}; Label Tag: {label_tag}"
        )

        start = time.time()
        if max_batch_size > 0:
            image_ids = image_ids[:max_batch_size]

        multi_gpu = request.get("multi_gpu", True)
        multi_gpus = request.get("gpus", "all")
        gpus = (
            list(range(torch.cuda.device_count())) if not multi_gpus or multi_gpus == "all" else multi_gpus.split(",")
        )
        device = name_to_device(request.get("device", "cuda"))
        device_ids = [f"cuda:{id}" for id in gpus] if multi_gpu else [device]

        result: dict = {}
        infer_tasks = []
        for idx, image_id in enumerate(image_ids):
            req = copy.deepcopy(request)
            req["_id"] = idx
            req["_image_id"] = image_id

            req["image"] = image_id
            req["save_label"] = True
            req["label_tag"] = label_tag
            req["logging"] = request.get("logging", "INFO")
            req["device"] = device_ids[idx % len(device_ids)]

            infer_tasks.append(req)
            result[image_id] = None

        total = len(infer_tasks)
        max_workers = request.get("max_workers", 0)
        max_workers = max_workers if max_workers else max(1, multiprocessing.cpu_count() // 4)
        max_workers = min(max_workers, multiprocessing.cpu_count())

        if len(infer_tasks) > 1 and (max_workers == 0 or max_workers > 1):
            logger.info(f"MultiGpu: {multi_gpu}; Using Device(s): {device_ids}; Max Workers: {max_workers}")
            futures = {}
            with ThreadPoolExecutor(max_workers if max_workers else None, "WSI Infer") as executor:
                for t in infer_tasks:
                    futures[t["_id"]] = t, executor.submit(run_infer_task, t, datastore, infer)

                for tid, (t, future) in futures.items():
                    image_id = t["_image_id"]
                    try:
                        res = future.result()
                        result[image_id] = res
                    except Exception:
                        logger.warning(f"Failed to finish Infer Task: {tid} => {image_id}", exc_info=True)
                        result[image_id] = {}

                    finished = len([a for a in result.values() if a is not None])
                    logger.info(f"{tid} => {image_id} => {t['device']} => {finished} / {total}")
        else:
            for t in infer_tasks:
                tid = t["_id"]
                res = run_infer_task(t, datastore, infer)

                image_id = t["_image_id"]
                result[image_id] = res
                finished = tid + 1
                logger.info(f"{tid} => {image_id} => {t['device']} => {finished} / {total}")

        latency_total = time.time() - start
        logger.info(f"Batch Infer Time Taken: {latency_total:.4f}")
        return result


def run_infer_task(req, datastore, infer):
    handle_torch_linalg_multithread(req)

    image_id = req.get("image")
    logger.info(f"Running inference for image id {image_id}")
    r = infer(req, datastore)
    if r.get("file"):
        remove_file(r.get("file"))
        r.pop("file")
    return r
