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

import glob
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Union

from monai.bundle import ConfigItem, ConfigParser
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Compose, LoadImaged, SaveImaged

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored
from monailabel.transform.pre import LoadImageTensord
from monailabel.utils.others.class_utils import unload_module
from monailabel.utils.others.generic import strtobool

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

    def key_detector(self) -> Sequence[str]:
        return ["detector"]

    def key_detector_ops(self) -> Sequence[str]:
        return ["detector_ops"]

    def key_displayable_configs(self) -> Sequence[str]:
        return ["displayable_configs"]


class BundleInferTask(BasicInferTask):
    """
    This provides Inference Engine for Monai Bundle.
    """

    def __init__(
        self,
        path: str,
        conf: Dict[str, str],
        const: Optional[BundleConstants] = None,
        type: Union[str, InferType] = "",
        pre_filter: Optional[Sequence] = None,
        post_filter: Optional[Sequence] = [SaveImaged],
        extend_load_image: bool = True,
        add_post_restore: bool = True,
        dropout: float = 0.0,
        load_strict=False,
        **kwargs,
    ):
        self.valid: bool = False
        self.const = const if const else BundleConstants()

        self.pre_filter = pre_filter
        self.post_filter = post_filter
        self.extend_load_image = extend_load_image
        self.dropout = dropout

        config_paths = [c for c in self.const.configs() if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no infer config {self.const.configs()} exists")
            return

        sys.path.insert(0, path)
        unload_module("scripts")

        self.bundle_path = path
        self.bundle_config_path = os.path.join(path, "configs", config_paths[0])
        self.bundle_config = self._load_bundle_config(self.bundle_path, self.bundle_config_path)
        # For deepedit inferer - allow the use of clicks
        self.bundle_config.config["use_click"] = True if type.lower() == "deepedit" else False

        if self.dropout > 0:
            self.bundle_config["network_def"]["dropout"] = self.dropout

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
                sys.path.remove(self.bundle_path)
                return

        # https://docs.monai.io/en/latest/mb_specification.html#metadata-json-file
        with open(os.path.join(path, "configs", self.const.metadata_json())) as fp:
            metadata = json.load(fp)

        self.key_image, image = next(iter(metadata["network_data_format"]["inputs"].items()))
        self.key_pred, pred = next(iter(metadata["network_data_format"]["outputs"].items()))

        # labels = ({v.lower(): int(k) for k, v in pred.get("channel_def", {}).items() if v.lower() != "background"})
        labels = {}
        for k, v in pred.get("channel_def", {}).items():
            if (not type.lower() == "deepedit") and (v.lower() != "background"):
                labels[v.lower()] = int(k)
            else:
                labels[v.lower()] = int(k)
        description = metadata.get("description")
        spatial_shape = image.get("spatial_shape")
        dimension = len(spatial_shape) if spatial_shape else 3
        type = self._get_type(os.path.basename(path), type)

        # if detection task, set post restore to False by default.
        self.add_post_restore = False if type == "detection" else add_post_restore

        super().__init__(
            path=model_path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            preload=strtobool(conf.get("preload", "false")),
            load_strict=load_strict,
            **kwargs,
        )

        # Add models options if more than one model is provided by bundle.
        pytorch_models = [os.path.basename(p) for p in glob.glob(os.path.join(path, "models", "*.pt"))]
        pytorch_models.sort(key=len)
        self._config.update({"model_filename": pytorch_models})
        # Add bundle's loadable params to MONAI Label config, load exposed keys and params to options panel
        for k in self.const.key_displayable_configs():
            if self.bundle_config.get(k):
                self.displayable_configs = self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
                self._config.update(self.displayable_configs)

        self.valid = True
        self.version = metadata.get("version")
        sys.path.remove(self.bundle_path)

    def is_valid(self) -> bool:
        return self.valid

    def info(self) -> Dict[str, Any]:
        i = super().info()
        i["version"] = self.version
        return i

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        # Update bundle parameters based on user's option
        for k in self.const.key_displayable_configs():
            if self.bundle_config.get(k):
                self.bundle_config[k].update({c: data[c] for c in self.displayable_configs.keys()})
                self.bundle_config.parse()

        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        pre = []
        for k in self.const.key_preprocessing():
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                pre = list(c.transforms) if isinstance(c, Compose) else c

        pre = self._filter_transforms(pre, self.pre_filter)

        for t in pre:
            if isinstance(t, LoadImaged):
                t._loader.image_only = False

        if pre and self.extend_load_image:
            res = []
            for t in pre:
                if isinstance(t, LoadImaged):
                    res.append(LoadImageTensord(keys=t.keys, load_image_d=t))
                else:
                    res.append(t)
            pre = res

        sys.path.remove(self.bundle_path)
        return pre

    def inferer(self, data=None) -> Inferer:
        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        i = None
        for k in self.const.key_inferer():
            if self.bundle_config.get(k):
                i = self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
                break

        sys.path.remove(self.bundle_path)
        return i if i is not None else SimpleInferer()

    def detector(self, data=None) -> Optional[Callable]:
        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        d = None
        for k in self.const.key_detector():
            if self.bundle_config.get(k):
                detector = self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
                for k in self.const.key_detector_ops():
                    self.bundle_config.get_parsed_content(k, instantiate=True)

                if detector is None or callable(detector):
                    d = detector  # type: ignore
                    break
                raise ValueError("Invalid Detector type;  It's not callable")

        sys.path.remove(self.bundle_path)
        return d

    def post_transforms(self, data=None) -> Sequence[Callable]:
        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        post = []
        for k in self.const.key_postprocessing():
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                post = list(c.transforms) if isinstance(c, Compose) else c

        post = self._filter_transforms(post, self.post_filter)

        if self.add_post_restore:
            post.append(Restored(keys=self.key_pred, ref_image=self.key_image))

        sys.path.remove(self.bundle_path)
        return post

    def _get_type(self, name, type):
        name = name.lower() if name else ""
        return (
            (
                InferType.DEEPEDIT
                if "deepedit" in name
                else InferType.DEEPGROW
                if "deepgrow" in name
                else InferType.DETECTION
                if "detection" in name
                else InferType.SEGMENTATION
                if "segmentation" in name
                else InferType.CLASSIFICATION
                if "classification" in name
                else InferType.SEGMENTATION
            )
            if not type
            else type
        )

    def _filter_transforms(self, transforms, filters):
        if not filters or not transforms:
            return transforms

        res = []
        for t in transforms:
            if not [f for f in filters if isinstance(t, f)]:
                res.append(t)
        return res

    def _update_device(self, data):
        k_device = self.const.key_device()
        device = data.get(k_device) if data else None
        if device:
            self.bundle_config.config.update({k_device: device})  # type: ignore
            if self.bundle_config.ref_resolver.items.get(k_device):
                self.bundle_config.ref_resolver.items[k_device] = ConfigItem(config=device, id=k_device)

    def _load_bundle_config(self, path, config):
        bundle_config = ConfigParser()
        bundle_config.read_config(config)
        bundle_config.config.update({self.const.key_bundle_root(): path})  # type: ignore
        return bundle_config
