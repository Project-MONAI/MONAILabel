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
import time
from typing import Any, Dict

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

logger = logging.getLogger(__name__)


class Epistemic(Strategy):
    """
    Epistemic as active learning strategy
    """

    SECS_IN_DAY = 24 * 60 * 60

    def __init__(
        self, k=0, reset=SECS_IN_DAY, key="epistemic_entropy", desc="Get First Sample Based on Epistemic score"
    ):
        self.k = k
        self.reset = reset  # Reset previously served samples after N seconds (ex: every day)
        self.key = key
        super().__init__(desc)

    def __call__(self, request, datastore: Datastore):
        images = datastore.get_unlabeled_images()

        if not len(images):
            return None

        scores: Dict[str, Any] = {}
        current_ts = int(time.time())
        strategy = request["strategy"]

        for image in images:
            info = datastore.get_image_info(image)
            score = info.get(self.key, 0)
            ts = min(current_ts - info.get("strategy", {}).get(strategy, {}).get("ts", 0), self.reset)
            scores[image] = {"score": score, "ts": ts}

        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)}  # type: ignore
        logger.info(f"{strategy}: Top-N: {scores}")

        # Pick Top-N based on epistemic scores
        top_k: Dict[str, Any] = {}
        max_len = self.k if 0 < self.k < len(scores) else len(scores)
        for k, v in scores.items():
            if len(top_k) == max_len:
                break
            # Handle similar timestamps
            top_k[k] = {
                "score": v["score"],
                "ts": v["ts"] - (pow(10, len(top_k)) if v["ts"] == self.reset else len(top_k)) * 10,
            }
        logger.info(f"{strategy}: Top-K: {top_k}")

        # Pick the one which is least served recently among Top-N
        top_k = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1]["ts"], reverse=True)}  # type: ignore
        logger.info(f"{strategy}: Top-K (ts): {top_k};")

        image = next(iter(top_k))
        logger.info(f"{strategy}: Selected Image: {image}; epistemic_entropy: {top_k[image]}")
        return image
