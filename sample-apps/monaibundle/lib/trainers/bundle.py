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

import monai.bundle
import torch
from monai.bundle import ConfigParser
from monai.data import partition_dataset
from monai.handlers import CheckpointLoader

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class BundleTrainTask(TrainTask):
    def __init__(self, path):
        self.bundle_config = None

        self.bundle_config_path = os.path.join(path, "configs", "train.json")
        if not os.path.exists(self.bundle_config_path):
            self.bundle_config_path = os.path.join(path, "configs", "train.yaml")
        if not os.path.exists(self.bundle_config_path):
            logger.warning(f"Ignore {path} as there is no infer/train config exists")
            return

        self.bundle_path = path
        self.bundle_config = ConfigParser()
        self.bundle_config.read_config(self.bundle_config_path)

        self.bundle_metadata = os.path.join(path, "configs", "metadata.json")
        with open(self.bundle_metadata) as fp:
            self.metadata = json.load(fp)

        super().__init__(self.metadata["description"])

    def is_valid(self):
        return True if self.bundle_config else False

    def config(self):
        return {
            "device": "cuda",
            "pretrained": True,
            "max_epochs": 10,
            "val_split": 0.2,
            "multi_gpu": True,
            "gpus": "all",
        }

    def _partition_datalist(self, datalist, request, shuffle=False):
        for d in datalist:
            d.pop("meta", None)

        val_split = request.get("val_split", 0.0)
        if val_split > 0.0:
            train_datalist, val_datalist = partition_dataset(
                datalist, ratios=[(1 - val_split), val_split], shuffle=shuffle
            )
        else:
            train_datalist = datalist
            val_datalist = []

        logger.info(f"Total Records for Training: {len(train_datalist)}")
        logger.info(f"Total Records for Validation: {len(val_datalist)}")
        return train_datalist, val_datalist

    def _device(self, str):
        return torch.device(str if torch.cuda.is_available() else "cpu")

    def _load_checkpoint(self, output_dir, pretrained, multi_gpu, device, train_handlers):
        load_path = os.path.join(output_dir, "model.pt") if pretrained else None
        if os.path.exists(load_path):
            logger.info(f"Add Checkpoint Loader for Path: {load_path}")

            map_location = {"cuda:0": f"cuda:{device.index}" if device.index else "cuda"} if multi_gpu else None
            network = self.bundle_config.get_parsed_content("network_def", instantiate=True)
            load_dict = {"model": network}
            train_handlers.append(CheckpointLoader(load_path, load_dict, map_location=map_location, strict=False))

    def __call__(self, request, datastore: Datastore):
        ds = datastore.datalist()
        train_ds, val_ds = self._partition_datalist(ds, request)

        max_epochs = request.get("max_epochs", 50)
        train_handlers = self.bundle_config.get("train#handlers", [])
        output_dir = os.path.join(self.bundle_path, "models")
        pretrained = request.get("pretrained", True)
        multi_gpu = request.get("multi_gpu", False)
        device = self._device(request.get("device", "cuda"))

        logger.info(f"Using device: {device}")
        self._load_checkpoint(output_dir, pretrained, multi_gpu, device, train_handlers)

        overrides = {
            "bundle_root": self.bundle_path,
            "train#trainer#max_epochs": max_epochs,
            "train#dataset#data": train_ds,
            "validate#dataset#data": val_ds,
            "device": device,
            "train#handlers": train_handlers,
        }

        monai.bundle.run("training", meta_file=self.bundle_metadata, config_file=self.bundle_config_path, **overrides)
        logger.info("Training Finished....")
        return {}
