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
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Compose, SaveImaged

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored

logger = logging.getLogger(__name__)


class Const:
    CONFIGS = ["inference.json", "inference.yaml"]
    METADATA_JSON = "metadata.json"
    MODEL_PYTORCH = "model.pt"
    MODEL_TORCHSCRIPT = "model.ts"

    KEY_DEVICE = "device"
    KEY_BUNDLE_ROOT = "bundle_root"
    KEY_NETWORK_DEF = "network_def"
    KEY_PREPROCESSING = ["preprocessing", "pre_transforms"]
    KEY_POSTPROCESSING = ["postprocessing", "post_transforms"]
    KEY_INFERER = ["inferer"]


class BundleInferTask(InferTask):
    """
    This provides Inference Engine for Monai Bundle.
    """

    def __init__(self, path, conf):
        self.valid: bool = False
        config_paths = [c for c in Const.CONFIGS if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no infer config {Const.CONFIGS} exists")
            return

        self.bundle_config = ConfigParser()
        self.bundle_config.read_config(os.path.join(path, "configs", config_paths[0]))
        self.bundle_config.config.update({Const.KEY_BUNDLE_ROOT: path})

        network = None
        model_path = os.path.join(path, "models", Const.MODEL_PYTORCH)
        if os.path.exists(model_path):
            network = self.bundle_config.get_parsed_content(Const.KEY_NETWORK_DEF, instantiate=True)
        else:
            model_path = os.path.join(path, "models", Const.MODEL_TORCHSCRIPT)
            if not os.path.exists(model_path):
                logger.warning(f"Ignore {path} as neither {Const.MODEL_PYTORCH} nor {Const.MODEL_TORCHSCRIPT} exists")
                return

        # https://docs.monai.io/en/latest/mb_specification.html#metadata-json-file
        with open(os.path.join(path, "configs", Const.METADATA_JSON)) as fp:
            metadata = json.load(fp)

        self.key_image, image = next(iter(metadata["network_data_format"]["inputs"].items()))
        self.key_pred, pred = next(iter(metadata["network_data_format"]["outputs"].items()))

        labels = {v.lower(): int(k) for k, v in pred.get("channel_def", {}).items() if v.lower() != "background"}
        description = metadata.get("description")
        spatial_shape = image.get("spatial_shape")
        dimension = len(spatial_shape) if spatial_shape else 3
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
            preload=conf.get("preload", False),
        )
        self.valid = True

    def is_valid(self) -> bool:
        return self.valid

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        pre = []
        for k in Const.KEY_PREPROCESSING:
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                pre = [t for t in c.transforms] if isinstance(c, Compose) else c
        return pre

    def inferer(self, data=None) -> Inferer:
        for k in Const.KEY_INFERER:
            if self.bundle_config.get(k):
                return self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        post = []
        for k in Const.KEY_POSTPROCESSING:
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                post = [t for t in c.transforms] if isinstance(c, Compose) else c

        post = [t for t in post if not isinstance(t, SaveImaged)]
        post.append(Restored(keys=self.key_pred, ref_image=self.key_image))
        return post
