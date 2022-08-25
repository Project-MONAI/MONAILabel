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
import subprocess

import monai.bundle
import torch
from monai.bundle import ConfigParser
from monai.data import partition_dataset
from monai.handlers import CheckpointLoader

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class Const:
    CONFIGS = ("train.json", "train.yaml")
    MULTI_GPU_CONFIGS = ("multi_gpu_train.json", "multi_gpu_train.yaml")
    METADATA_JSON = "metadata.json"
    MODEL_PYTORCH = "model.pt"

    KEY_DEVICE = "device"
    KEY_BUNDLE_ROOT = "bundle_root"
    KEY_NETWORK = "network"
    KEY_NETWORK_DEF = "network_def"
    KEY_TRAIN_TRAINER_MAX_EPOCHS = "train#trainer#max_epochs"
    KEY_TRAIN_DATASET_DATA = "train#dataset#data"
    KEY_TRAIN_HANDLERS = "train#handlers"
    KEY_VALIDATE_DATASET_DATA = "validate#dataset#data"


class BundleTrainTask(TrainTask):
    def __init__(self, path, conf):
        self.valid: bool = False
        config_paths = [c for c in Const.CONFIGS if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no train config {Const.CONFIGS} exists")
            return

        self.bundle_path = path
        self.bundle_config_path = os.path.join(path, "configs", config_paths[0])

        self.bundle_config = ConfigParser()
        self.bundle_config.read_config(self.bundle_config_path)
        self.bundle_config.config.update({Const.KEY_BUNDLE_ROOT: self.bundle_path})

        # https://docs.monai.io/en/latest/mb_specification.html#metadata-json-file
        self.bundle_metadata_path = os.path.join(path, "configs", "metadata.json")
        with open(os.path.join(path, "configs", Const.METADATA_JSON)) as fp:
            metadata = json.load(fp)

        super().__init__(metadata.get("description", ""))
        self.valid = True

    def is_valid(self):
        return self.valid

    def config(self):
        return {
            "device": "cuda",  # DEVICE
            "pretrained": True,  # USE EXISTING CHECKPOINT/PRETRAINED MODEL
            "max_epochs": 50,  # TOTAL EPOCHS TO RUN
            "val_split": 0.2,  # VALIDATION SPLIT; -1 TO USE DEFAULT FROM BUNDLE
            "multi_gpu": True,  # USE MULTI-GPU
            "gpus": "all",  # COMMA SEPARATE DEVICE INDEX
        }

    def _partition_datalist(self, datalist, request, shuffle=False):
        # only use image and label attributes; skip for other meta info from datastore for now
        datalist = [{"image": d["image"], "label": d["label"]} for d in datalist if d]
        logger.info(f"Total Records in Dataset: {len(datalist)}")

        val_split = request.get("val_split", 0.0)
        if val_split > 0.0:
            train_datalist, val_datalist = partition_dataset(
                datalist, ratios=[(1 - val_split), val_split], shuffle=shuffle
            )
        else:
            train_datalist = datalist
            val_datalist = None if val_split < 0 else []

        logger.info(f"Total Records for Training: {len(train_datalist)}")
        logger.info(f"Total Records for Validation: {len(val_datalist) if val_datalist else ''}")
        return train_datalist, val_datalist

    def _device(self, str):
        return torch.device(str if torch.cuda.is_available() else "cpu")

    def _load_checkpoint(self, output_dir, pretrained, train_handlers):
        load_path = os.path.join(output_dir, Const.MODEL_PYTORCH) if pretrained else None
        if os.path.exists(load_path):
            logger.info(f"Add Checkpoint Loader for Path: {load_path}")

            load_dict = {"model": f"$@{Const.KEY_NETWORK}"}
            if not [t for t in train_handlers if t.get("_target_") == CheckpointLoader.__name__]:
                loader = {
                    "_target_": CheckpointLoader.__name__,
                    "load_path": load_path,
                    "load_dict": load_dict,
                    "strict": False,
                }
                train_handlers.insert(0, loader)

    def __call__(self, request, datastore: Datastore):
        ds = datastore.datalist()
        train_ds, val_ds = self._partition_datalist(ds, request)

        max_epochs = request.get("max_epochs", 50)
        pretrained = request.get("pretrained", True)
        multi_gpu = request.get("multi_gpu", False)
        multi_gpu = multi_gpu if torch.cuda.device_count() > 1 else False

        gpus = request.get("gpus", "all")
        gpus = list(range(torch.cuda.device_count())) if gpus == "all" else [int(g) for g in gpus.split(",")]
        logger.info(f"Using Multi GPU: {multi_gpu}; GPUS: {gpus}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        device = self._device(request.get("device", "cuda"))
        logger.info(f"Using device: {device}")

        train_handlers = self.bundle_config.get(Const.KEY_TRAIN_HANDLERS, [])
        self._load_checkpoint(os.path.join(self.bundle_path, "models"), pretrained, train_handlers)

        overrides = {
            Const.KEY_BUNDLE_ROOT: self.bundle_path,
            Const.KEY_TRAIN_TRAINER_MAX_EPOCHS: max_epochs,
            Const.KEY_TRAIN_DATASET_DATA: train_ds,
            Const.KEY_DEVICE: device,
            Const.KEY_TRAIN_HANDLERS: train_handlers,
        }

        # external validation datalist supported through bundle itself (pass -1 in the request to use the same)
        if val_ds is not None:
            overrides[Const.KEY_VALIDATE_DATASET_DATA] = val_ds

        if multi_gpu:
            config_paths = [
                c for c in Const.MULTI_GPU_CONFIGS if os.path.exists(os.path.join(self.bundle_path, "configs", c))
            ]
            if not config_paths:
                logger.warning(f"Ignore Multi-GPU Training; No multi-gpu train config {Const.MULTI_GPU_CONFIGS} exists")
                return

            train_path = os.path.join(self.bundle_path, "configs", "monailabel_train.json")
            multi_gpu_train_path = os.path.join(self.bundle_path, "configs", config_paths[0])
            logging_file = os.path.join(self.bundle_path, "configs", "logging.conf")
            for k, v in overrides.items():
                if k != Const.KEY_DEVICE:
                    self.bundle_config.set(v, k)
            ConfigParser.export_config_file(self.bundle_config.config, train_path, indent=2)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpus])
            logger.info(f"Using CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
            cmd = [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={len(gpus)}",
                "-m",
                "monai.bundle",
                "run",
                "training",
                "--meta_file",
                self.bundle_metadata_path,
                "--config_file",
                f"['{train_path}','{multi_gpu_train_path}']",
                "--logging_file",
                logging_file,
            ]
            self.run_command(cmd, env)
        else:
            monai.bundle.run(
                "training",
                meta_file=self.bundle_metadata_path,
                config_file=self.bundle_config_path,
                **overrides,
            )

        logger.info("Training Finished....")
        return {}

    def run_command(self, cmd, env):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, env=env)
        while process.poll() is None:
            line = process.stdout.readline()
            line = line.rstrip()
            if line:
                print(line, flush=True)

        logger.info(f"Return code: {process.returncode}")
        process.stdout.close()
