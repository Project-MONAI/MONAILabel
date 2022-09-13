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
from monai.metrics.active_learning_metrics import VarianceMetric

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
        max_samples=0,
        simulation_size=5,
        use_variance=False,
        key_output_entropy="epistemic_entropy",
        key_output_ts="epistemic_ts",
    ):
        super().__init__(f"Compute initial score based on dropout - {infer_task.description}")
        self.infer_task = infer_task
        self.dimension = infer_task.dimension

        self.max_samples = max_samples
        self.simulation_size = simulation_size
        self.use_variance = use_variance
        self.key_output_entropy = key_output_entropy
        self.key_output_ts = key_output_ts

    def entropy_volume(self, vol_input):
        # The input is assumed with repetitions, channels and then volumetric data
        vol_input = vol_input.cpu().detach().numpy() if isinstance(vol_input, torch.Tensor) else vol_input
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

    def variance_volume(self, vol_input, ignore_nans=True):
        if ignore_nans:
            vol_input[vol_input <= 0] = 0.0005
            vari = np.nanvar(vol_input.cpu().detach().numpy(), axis=0)  # torch.var vs np.nanvar (ignore_nans)
            variance = np.sum(vari, axis=0)
        else:
            variance_metric = VarianceMetric(threshold=0.0005, spatial_map=True, scalar_reduction="sum")
            variance = variance_metric(vol_input)

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
        max_samples = request.get("max_samples", self.max_samples)
        simulation_size = request.get("simulation_size", self.simulation_size)
        if simulation_size < 2:
            simulation_size = 2
            logger.warning("EPISTEMIC:: Fixing 'simulation_size=2' as min 2 simulations are needed to compute entropy")

        logger.info(f"EPISTEMIC:: Total unlabeled images: {len(unlabeled_images)}; max_samples: {max_samples}")
        t_start = time.time()

        image_ids = []
        for image_id in unlabeled_images:
            image_info = datastore.get_image_info(image_id)
            prev_ts = image_info.get("epistemic_ts", 0)
            if prev_ts == model_ts:
                skipped += 1
                continue
            image_ids.append(image_id)
        image_ids = image_ids[:max_samples] if max_samples else image_ids

        max_workers = request.get("max_workers", 2)
        multi_gpu = request.get("multi_gpu", False)
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
                    futures.append(e.submit(self.run_scoring, image_id, simulation_size, model_ts, datastore))
                for future in futures:
                    future.result()
        else:
            for image_id in image_ids:
                self.run_scoring(image_id, simulation_size, model_ts, datastore)

        summary = {
            "total": len(unlabeled_images),
            "skipped": skipped,
            "executed": len(image_ids),
            "latency": round(time.time() - t_start, 3),
        }

        logger.info(f"EPISTEMIC:: {summary}")
        self.infer_task.clear_cache()
        return summary

    def run_scoring(self, image_id, simulation_size, model_ts, datastore):
        start = time.time()
        request = {
            "image": datastore.get_image_uri(image_id),
            "logging": "error",
            "cache_transforms": False,
        }

        accum_unl_outputs = []
        for i in range(simulation_size):
            data = self.infer_task(request=request)
            pred = data[self.infer_task.output_label_key] if isinstance(data, dict) else None
            if pred is not None:
                logger.debug(f"EPISTEMIC:: {image_id} => {i} => pred: {pred.shape}; sum: {np.sum(pred)}")
                accum_unl_outputs.append(pred)
            else:
                logger.info(f"EPISTEMIC:: {image_id} => {i} => pred: None")

        accum = torch.stack(accum_unl_outputs)
        accum = torch.squeeze(accum)

        # Accum Expected shape for 2D images is (N, C, H, W) for 3D (N, C, H, W, D)
        # To handle cases where only a single class of segmentation is present, an extra dimension is added
        if self.dimension == 2 and len(accum.shape) == 3:
            accum = torch.unsqueeze(accum, dim=1)
        elif self.dimension == 3 and len(accum.shape) == 4:
            accum = torch.unsqueeze(accum, dim=1)

        entropy = self.variance_volume(accum) if self.use_variance else self.entropy_volume(accum)
        entropy = float(np.nanmean(entropy))

        latency = time.time() - start
        logger.info(
            "EPISTEMIC:: {} => iters: {}; entropy: {}; latency: {};".format(
                image_id,
                simulation_size,
                round(entropy, 4),
                round(latency, 3),
            )
        )

        # Add epistemic_entropy in datastore
        info = {self.key_output_entropy: entropy, self.key_output_ts: model_ts}
        datastore.update_image_info(image_id, info)
