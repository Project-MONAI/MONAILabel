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
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod

logger = logging.getLogger(__name__)


class EpistemicScoring(ScoringMethod):
    """
    First version of Epistemic computation used as active learning strategy
    """

    def __init__(
        self,
        infer_task: InferTask,
        num_samples=10,
        use_variance=False,
    ):
        super().__init__(f"Compute initial score based on dropout - {infer_task.description}")
        self.infer_task = infer_task
        self.dimension = infer_task.dimension

        self.num_samples = num_samples
        self.use_variance = use_variance

    def entropy_volume(self, vol_input):
        # The input is assumed with repetitions, channels and then volumetric data
        vol_input = vol_input.astype(dtype="float32")
        dims = vol_input.shape
        reps = dims[0]
        entropy = np.zeros(dims[2:], dtype="float32")

        # Threshold values less than or equal to zero
        threshold = 0.00005
        vol_input[vol_input <= 0] = threshold

        # Looping across channels as each channel is a class
        if len(dims) == 5 if self.dimension == 3 else 4:
            for channel in range(dims[1]):
                t_vol = np.squeeze(
                    vol_input[:, channel, :, :, :] if self.dimension == 3 else vol_input[:, channel, :, :]
                )
                t_sum = np.sum(t_vol, axis=0)
                t_avg = np.divide(t_sum, reps)
                t_log = np.log(t_avg)
                t_entropy = -np.multiply(t_avg, t_log)
                entropy = entropy + t_entropy
        else:
            t_vol = np.squeeze(vol_input)
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            t_entropy = -np.multiply(t_avg, t_log)
            entropy = entropy + t_entropy

        # Returns a 3D volume of entropy
        return entropy

    def variance_volume(self, vol_input):
        vol_input = vol_input.astype(dtype="float32")

        # Threshold values less than or equal to zero
        threshold = 0.0005
        vol_input[vol_input <= 0] = threshold

        vari = np.nanvar(vol_input, axis=0)
        variance = np.sum(vari, axis=0)

        if self.dimension == 3:
            variance = np.expand_dims(variance, axis=0)
            variance = np.expand_dims(variance, axis=0)
        return variance

    def __call__(self, request, datastore: Datastore):
        logger.info("Starting Epistemic Uncertainty scoring")

        model_file = self.infer_task.get_path()
        model_ts = int(os.stat(model_file).st_mtime) if model_file and os.path.exists(model_file) else 1
        self.infer_task.clear_cache()

        # Performing Epistemic for all unlabeled images
        skipped = 0
        unlabeled_images = datastore.get_unlabeled_images()
        num_samples = request.get("num_samples", self.num_samples)
        if num_samples < 2:
            num_samples = 2
            logger.warning("EPISTEMIC:: Fixing 'num_samples=2' as min 2 samples are needed to compute entropy")

        logger.info(f"EPISTEMIC:: Total unlabeled images: {len(unlabeled_images)}")
        t_start = time.time()

        image_ids = []
        for image_id in unlabeled_images:
            image_info = datastore.get_image_info(image_id)
            prev_ts = image_info.get("epistemic_ts", 0)
            if prev_ts == model_ts:
                skipped += 1
                continue
            image_ids.append(image_id)

        max_workers = request.get("max_workers", 2)
        multi_gpu = request.get("multi_gpu", True)
        multi_gpus = request.get("gpus", "all")
        gpus = (
            list(range(torch.cuda.device_count())) if not multi_gpus or multi_gpus == "all" else multi_gpus.split(",")
        )
        device_ids = [f"cuda:{id}" for id in gpus] if multi_gpu else [request.get("device", "cuda")]

        max_workers = max_workers if max_workers else max(1, multiprocessing.cpu_count() // 2)
        max_workers = min(max_workers, multiprocessing.cpu_count())

        if len(image_ids) > 1 and (max_workers == 0 or max_workers > 1):
            logger.info(f"MultiGpu: {multi_gpu}; Using Device(s): {device_ids}; Max Workers: {max_workers}")
            futures = []
            with ThreadPoolExecutor(max_workers if max_workers else None, "ScoreInfer") as e:
                for image_id in image_ids:
                    futures.append(e.submit(self.run_scoring, image_id, num_samples, model_ts, datastore))
                for future in futures:
                    future.result()
        else:
            for image_id in image_ids:
                self.run_scoring(image_id, num_samples, model_ts, datastore)

        summary = {
            "total": len(unlabeled_images),
            "skipped": skipped,
            "executed": len(image_ids),
            "latency": round(time.time() - t_start, 3),
        }

        logger.info(f"EPISTEMIC:: {summary}")
        self.infer_task.clear_cache()
        return summary

    def run_scoring(self, image_id, num_samples, model_ts, datastore):
        start = time.time()
        request = {
            "image": datastore.get_image_uri(image_id),
            "logging": "error",
            "cache_transforms": False,
        }

        accum_unl_outputs = []
        for i in range(num_samples):
            data = self.infer_task(request=request)
            pred = data[self.infer_task.output_label_key] if isinstance(data, dict) else None
            if pred is not None:
                logger.debug(f"EPISTEMIC:: {image_id} => {i} => pred: {pred.shape}; sum: {np.sum(pred)}")
                accum_unl_outputs.append(pred)
            else:
                logger.info(f"EPISTEMIC:: {image_id} => {i} => pred: None")

        accum_numpy = np.stack(accum_unl_outputs)
        accum_numpy = np.squeeze(accum_numpy)
        if self.dimension == 3:
            accum_numpy = accum_numpy[:, 1:, :, :, :] if len(accum_numpy.shape) > 4 else accum_numpy
        else:
            accum_numpy = accum_numpy[:, 1:, :, :] if len(accum_numpy.shape) > 3 else accum_numpy

        entropy = self.variance_volume(accum_numpy) if self.use_variance else self.entropy_volume(accum_numpy)
        entropy = float(np.nanmean(entropy))

        latency = time.time() - start
        logger.info(
            "EPISTEMIC:: {} => iters: {}; entropy: {}; latency: {};".format(
                image_id,
                num_samples,
                round(entropy, 4),
                round(latency, 3),
            )
        )

        # Add epistemic_entropy in datastore
        info = {"epistemic_entropy": entropy, "epistemic_ts": model_ts}
        datastore.update_image_info(image_id, info)
