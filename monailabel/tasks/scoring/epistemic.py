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
import time

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.scoring import ScoringMethod

logger = logging.getLogger(__name__)


class EpistemicScoring(ScoringMethod):
    """
    First version of Epistemic computation used as active learning strategy
    """

    def __init__(
        self, model, network=None, transforms=None, roi_size=(128, 128, 64), num_samples=10, load_strict=False
    ):
        super().__init__("Compute initial score based on dropout")
        self.model = model
        self.network = network
        self.transforms = transforms
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.roi_size = roi_size
        self.num_samples = num_samples
        self.load_strict = load_strict

    def infer_seg(self, data, model, roi_size, sw_batch_size):
        pre_transforms = (
            None
            if not self.transforms
            else self.transforms
            if isinstance(self.transforms, Compose)
            else Compose(self.transforms)
        )
        # data = run_transforms(data, pre_transforms, log_prefix="EPISTEMIC-PRE") if pre_transforms else data
        data = pre_transforms(data)

        with torch.no_grad():
            preds = sliding_window_inference(
                inputs=data["image"][None].to(self.device),
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
            )

        soft_preds = torch.softmax(preds, dim=1) if preds.shape[1] > 1 else torch.sigmoid(preds)
        soft_preds = soft_preds.detach().to("cpu").numpy()
        return soft_preds

    def entropy_3d_volume(self, vol_input):
        # The input is assumed with repetitions, channels and then volumetric data
        vol_input = vol_input.astype(dtype="float32")
        dims = vol_input.shape
        reps = dims[0]
        entropy = np.zeros(dims[2:], dtype="float32")

        # Threshold values less than or equal to zero
        threshold = 0.00005
        vol_input[vol_input <= 0] = threshold

        # Looping across channels as each channel is a class
        if len(dims) == 5:
            for channel in range(dims[1]):
                t_vol = np.squeeze(vol_input[:, channel, :, :, :])
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

    @staticmethod
    def _get_model_path(path):
        if not path:
            return None

        paths = [path] if isinstance(path, str) else path
        for path in reversed(paths):
            if os.path.exists(path):
                return path
        return None

    def _load_model(self, path, network):
        model_file = EpistemicScoring._get_model_path(path)
        if not model_file and not network:
            logger.warning(f"Skip Epistemic Scoring:: Model(s) {path} not available yet")
            return None, None

        logger.info(f"Using {model_file} for running Epistemic")
        model_ts = int(os.stat(model_file).st_mtime) if model_file and os.path.exists(model_file) else 1
        if network:
            model = copy.deepcopy(network)
            if model_file:
                if torch.cuda.is_available():
                    checkpoint = torch.load(model_file)
                else:
                    checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
                model_state_dict = checkpoint.get("model", checkpoint)
                model.load_state_dict(model_state_dict, strict=self.load_strict)
        else:
            if torch.cuda.is_available():
                model = torch.jit.load(model_file)
            else:
                model = torch.jit.load(model_file, map_location=torch.device("cpu"))
        return model, model_ts

    def __call__(self, request, datastore: Datastore):
        logger.info("Starting Epistemic Uncertainty scoring")

        result = {}
        model, model_ts = self._load_model(self.model, self.network)
        if not model:
            return
        model = model.to(self.device).train()

        # Performing Epistemic for all unlabeled images
        skipped = 0
        unlabeled_images = datastore.get_unlabeled_images()
        num_samples = request.get("num_samples", self.num_samples)
        if num_samples < 2:
            num_samples = 2
            logger.warning("EPISTEMIC:: Fixing 'num_samples=2' as min 2 samples are needed to compute entropy")

        logger.info(f"EPISTEMIC:: Total unlabeled images: {len(unlabeled_images)}")
        for image_id in unlabeled_images:
            image_info = datastore.get_image_info(image_id)
            prev_ts = image_info.get("epistemic_ts", 0)
            if prev_ts == model_ts:
                skipped += 1
                continue

            logger.info(f"EPISTEMIC:: Run for image: {image_id}; Prev Ts: {prev_ts}; New Ts: {model_ts}")

            # Computing the Entropy
            start = time.time()
            data = {"image": datastore.get_image_uri(image_id)}
            accum_unl_outputs = []
            for i in range(num_samples):
                output_pred = self.infer_seg(data, model, self.roi_size, 1)
                logger.info(f"EPISTEMIC:: {image_id} => {i} => pred: {output_pred.shape}; sum: {np.sum(output_pred)}")
                accum_unl_outputs.append(output_pred)

            accum_numpy = np.stack(accum_unl_outputs)
            accum_numpy = np.squeeze(accum_numpy)
            accum_numpy = accum_numpy[:, 1:, :, :, :] if len(accum_numpy.shape) > 4 else accum_numpy

            entropy = float(np.nanmean(self.entropy_3d_volume(accum_numpy)))

            if self.device == "cuda":
                torch.cuda.empty_cache()
            latency = time.time() - start

            logger.info(f"EPISTEMIC:: {image_id} => entropy: {entropy}")
            logger.info(f"EPISTEMIC:: Time taken for {num_samples} Monte Carlo Simulation samples: {latency}")

            # Add epistemic_entropy in datastore
            info = {"epistemic_entropy": entropy, "epistemic_ts": model_ts}
            datastore.update_image_info(image_id, info)
            result[image_id] = info

        logger.info(f"EPISTEMIC:: Total: {len(unlabeled_images)}; Skipped = {skipped}; Executed: {len(result)}")
        return result
