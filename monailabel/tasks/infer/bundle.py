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
from typing import Callable, Dict, Optional, Sequence

from monai.bundle import ConfigParser
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Compose, SaveImaged

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored

logger = logging.getLogger(__name__)


class BundleConstants:
    def configs(self) -> Sequence[str]:
        return ["inference.json", "inference.yaml"]

    def metadata_json(self) -> str:
        return "metadata.json"

    def model_pytorch(self) -> str:
        return "model.pt"

    def model_torchscript(self) -> str:
        return "model.ts"

    def key_device(self) -> str:
        return "device"

    def key_bundle_root(self) -> str:
        return "bundle_root"

    def key_network_def(self) -> str:
        return "network_def"

    def key_preprocessing(self) -> Sequence[str]:
        return ["preprocessing", "pre_transforms"]

    def key_postprocessing(self) -> Sequence[str]:
        return ["postprocessing", "post_transforms"]

    def key_inferer(self) -> Sequence[str]:
        return ["inferer"]


class BundleInferTask(InferTask):
    """
    This provides Inference Engine for Monai Bundle.
    """

    def __init__(
        self, path: str, conf: Dict[str, str], const: Optional[BundleConstants] = None, type: Optional[InferType] = None
    ):
        self.valid: bool = False
        self.const = const if const else BundleConstants()

        config_paths = [c for c in self.const.configs() if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no infer config {self.const.configs()} exists")
            return

        self.bundle_config = ConfigParser()
        self.bundle_config.read_config(os.path.join(path, "configs", config_paths[0]))
        self.bundle_config.config.update({self.const.key_bundle_root(): path})

        network = None
        model_path = os.path.join(path, "models", self.const.model_pytorch())
        if os.path.exists(model_path):
            network = self.bundle_config.get_parsed_content(self.const.key_network_def(), instantiate=True)
        else:
            model_path = os.path.join(path, "models", self.const.model_torchscript())
            if not os.path.exists(model_path):
                logger.warning(
                    f"Ignore {path} as neither {self.const.model_pytorch()} nor {self.const.model_torchscript()} exists"
                )
                return

        # https://docs.monai.io/en/latest/mb_specification.html#metadata-json-file
        with open(os.path.join(path, "configs", self.const.metadata_json())) as fp:
            metadata = json.load(fp)

        self.key_image, image = next(iter(metadata["network_data_format"]["inputs"].items()))
        self.key_pred, pred = next(iter(metadata["network_data_format"]["outputs"].items()))

        labels = {v.lower(): int(k) for k, v in pred.get("channel_def", {}).items() if v.lower() != "background"}
        description = metadata.get("description")
        spatial_shape = image.get("spatial_shape")
        dimension = len(spatial_shape) if spatial_shape else 3
        type = (
            (
                InferType.DEEPEDIT
                if "deepedit" in description.lower()
                else InferType.DEEPGROW
                if "deepgrow" in description.lower()
                else InferType.SEGMENTATION
            )
            if not type
            else type
        )

        super().__init__(
            path=model_path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            preload=conf.get("preload", False),
        )
        self.valid = True

    def is_valid(self) -> bool:
        return self.valid

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        pre = []
        for k in self.const.key_preprocessing():
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                pre = list(c.transforms) if isinstance(c, Compose) else c
        return pre

    def inferer(self, data=None) -> Inferer:
        for k in self.const.key_inferer():
            if self.bundle_config.get(k):
                return self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        post = []
        for k in self.const.key_postprocessing():
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                post = list(c.transforms) if isinstance(c, Compose) else c

        post = [t for t in post if not isinstance(t, SaveImaged)]
        post.append(Restored(keys=self.key_pred, ref_image=self.key_image))
        return post
