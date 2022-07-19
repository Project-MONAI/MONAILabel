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
import json
import logging
import os
from typing import Callable, Sequence

from monai.bundle import ConfigParser
from monai.inferers import Inferer
from monai.transforms import SaveImaged

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored

logger = logging.getLogger(__name__)


class BundleInferTask(InferTask):
    """
    This provides Inference Engine for Monai Bundle.
    """

    def __init__(self, path):
        self.bundle_config = None

        self.bundle_config_path = os.path.join(path, "configs", "inference.json")
        if not os.path.exists(self.bundle_config_path):
            self.bundle_config_path = os.path.join(path, "configs", "inference.yaml")
        if not os.path.exists(self.bundle_config_path):
            logger.warning(f"Ignore {path} as there is no infer config exists")
            return

        self.bundle_path = path
        self.bundle_config = ConfigParser()
        self.bundle_config.read_config(self.bundle_config_path)
        self.bundle_config.config.update({"bundle_root": self.bundle_path})

        network = None
        model_path = os.path.join(path, "models", "model.pt")
        if os.path.exists(model_path):
            network = self.bundle_config.get_parsed_content("network_def", instantiate=True)
        else:
            model_path = os.path.join(path, "models", "model.ts")
            if not os.path.exists(model_path):
                logger.warning(f"Ignore {path} as there is no model.ts or model.pt exists")
                return

        with open(os.path.join(path, "configs", "metadata.json")) as fp:
            self.metadata = json.load(fp)

        self.image_key, image = next(iter(self.metadata["network_data_format"]["inputs"].items()))
        self.pred_key, pred = next(iter(self.metadata["network_data_format"]["outputs"].items()))
        labels = {v: int(k) for k, v in pred["channel_def"].items()}
        labels.pop("background", None)

        description = self.metadata["description"]
        dimension = len(image["spatial_shape"])
        type = (
            InferType.DEEPEDIT
            if "deepedit" in description.lower()
            else InferType.DEEPGROW
            if "deepgrow" in description.lower()
            else InferType.SEGMENTATION
        )

        super().__init__(
            path=model_path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def is_valid(self) -> bool:
        return True if self.bundle_config is not None else False

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        try:
            pre = self.bundle_config.get_parsed_content("preprocessing", instantiate=True)
        except:
            pre = self.bundle_config.get_parsed_content("pre_transforms", instantiate=True)
        return [t for t in pre.transforms]

    def inferer(self, data=None) -> Inferer:
        infer: Inferer = self.bundle_config.get_parsed_content("inferer", instantiate=True)
        return infer

    def post_transforms(self, data=None) -> Sequence[Callable]:
        try:
            post = self.bundle_config.get_parsed_content("postprocessing", instantiate=True)
        except:
            post = self.bundle_config.get_parsed_content("post_transforms", instantiate=True)
        p = [t for t in post.transforms if not isinstance(t, SaveImaged)]
        p.append(Restored(keys=self.pred_key, ref_image=self.image_key))
        return p
