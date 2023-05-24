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
import subprocess
import sys
from typing import Dict, Optional, Sequence

import monai.bundle
import torch
from monai.bundle import ConfigParser
from monai.data import partition_dataset
from monai.handlers import CheckpointLoader

from monailabel.config import settings
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.class_utils import unload_module
from monailabel.utils.others.generic import device_list, name_to_device

logger = logging.getLogger(__name__)


class BundleConstants:
    def configs(self) -> Sequence[str]:
        return ["train.json", "train.yaml"]

    def multi_gpu_configs(self) -> Sequence[str]:
        return ["multi_gpu_train.json", "multi_gpu_train.yaml"]

    def metadata_json(self) -> str:
        return "metadata.json"

    def model_pytorch(self) -> str:
        return "model.pt"

    def key_device(self) -> str:
        return "device"

    def key_bundle_root(self) -> str:
        return "bundle_root"

    def key_network(self) -> str:
        return "network"

    def key_network_def(self) -> str:
        return "network_def"

    def key_train_trainer_max_epochs(self) -> str:
        return "train#trainer#max_epochs"

    def key_train_dataset_data(self) -> str:
        return "train#dataset#data"

    def key_train_handlers(self) -> str:
        return "train#handlers"

    def key_validate_dataset_data(self) -> str:
        return "validate#dataset#data"

    def key_tracking(self) -> str:
        return "tracking"

    def key_tracking_uri(self) -> str:
        return "tracking_uri"

    def key_experiment_name(self) -> str:
        return "experiment_name"

    def key_run_name(self) -> str:
        return "run_name"

    def key_displayable_configs(self) -> Sequence[str]:
        return ["displayable_configs"]


class BundleTrainTask(TrainTask):
    def __init__(
        self,
        path: str,
        conf: Dict[str, str],
        const: Optional[BundleConstants] = None,
        enable_tracking=False,
        model_dict_key="model",
        load_strict=False,
    ):
        self.valid: bool = False
        self.conf = conf
        self.const = const if const else BundleConstants()
        self.enable_tracking = enable_tracking
        self.model_dict_key = model_dict_key
        self.load_strict = load_strict

        config_paths = [c for c in self.const.configs() if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no train config {self.const.configs()} exists")
            return

        self.bundle_path = path
        self.bundle_config_path = os.path.join(path, "configs", config_paths[0])
        self.bundle_config = self._load_bundle_config(self.bundle_path, self.bundle_config_path)

        # https://docs.monai.io/en/latest/mb_specification.html#metadata-json-file
        self.bundle_metadata_path = os.path.join(path, "configs", "metadata.json")
        with open(os.path.join(path, "configs", self.const.metadata_json())) as fp:
            metadata = json.load(fp)

        super().__init__(metadata.get("description", ""))
        self.valid = True
        self.version = metadata.get("version")

    def is_valid(self):
        return self.valid

    def info(self):
        i = super().info()
        i["version"] = self.version
        return i

    def config(self):
        # Add models and param optiom to train option panel
        pytorch_models = [os.path.basename(p) for p in glob.glob(os.path.join(self.bundle_path, "models", "*.pt"))]
        pytorch_models.sort(key=len)

        config_options = {
            "device": device_list(),  # DEVICE
            "pretrained": True,  # USE EXISTING CHECKPOINT/PRETRAINED MODEL
            "max_epochs": 50,  # TOTAL EPOCHS TO RUN
            "val_split": 0.2,  # VALIDATION SPLIT; -1 TO USE DEFAULT FROM BUNDLE
            "multi_gpu": True,  # USE MULTI-GPU
            "gpus": "all",  # COMMA SEPARATE DEVICE INDEX
            "tracking": ["mlflow", "None"]
            if self.enable_tracking and settings.MONAI_LABEL_TRACKING_ENABLED
            else ["None", "mlflow"],
            "tracking_uri": settings.MONAI_LABEL_TRACKING_URI,
            "tracking_experiment_name": "",
            "run_id": "",  # bundle run id, if different from default
            "model_filename": pytorch_models,
        }

        for k in self.const.key_displayable_configs():
            if self.bundle_config.get(k):
                config_options.update(self.bundle_config.get_parsed_content(k, instantiate=True))  # type: ignore

        return config_options

    def _fetch_datalist(self, request, datastore: Datastore):
        datalist = datastore.datalist()

        # only use image and label attributes; skip for other meta info from datastore for now
        datalist = [{"image": d["image"], "label": d["label"]} for d in datalist if d]

        if "detection" in request.get("model"):
            # Generate datalist for detection task, box and label keys are used by default.
            # Future: either use box and label keys for all detection models, or set these keys by config.
            for idx, d in enumerate(datalist):
                with open(d["label"]) as fp:
                    json_object = json.loads(fp.read())  # load box coordinates from subject JSON
                    bboxes = [bdict["center"] + bdict["size"] for bdict in json_object["markups"]]

                # Only support detection, classification label do not suppot in bundle yet,
                # 0 is used for all positive boxes, wait for sync.
                datalist[idx] = {"image": d["image"], "box": bboxes, "label": [0] * len(bboxes)}

        return datalist

    def _partition_datalist(self, datalist, request, shuffle=False):
        val_split = request.get("val_split", 0.2)
        logger.info(f"Total Records in Dataset: {len(datalist)}; Validation Split: {val_split}")

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

    def _load_checkpoint(self, model_pytorch, pretrained, train_handlers):
        load_path = model_pytorch if pretrained else None
        if os.path.exists(load_path):
            logger.info(f"Add Checkpoint Loader for Path: {load_path}")

            load_dict = {self.model_dict_key: f"$@{self.const.key_network()}"}
            if not [t for t in train_handlers if t.get("_target_") == CheckpointLoader.__name__]:
                loader = {
                    "_target_": CheckpointLoader.__name__,
                    "load_path": load_path,
                    "load_dict": load_dict,
                    "strict": self.load_strict,
                }
                train_handlers.insert(0, loader)

    def __call__(self, request, datastore: Datastore):
        logger.info(f"Train Request: {request}")
        ds = self._fetch_datalist(request, datastore)
        train_ds, val_ds = self._partition_datalist(ds, request)

        max_epochs = request.get("max_epochs", 50)
        pretrained = request.get("pretrained", True)
        multi_gpu = request.get("multi_gpu", True)
        force_multi_gpu = request.get("force_multi_gpu", False)
        run_id = request.get("run_id", "run")

        multi_gpu = multi_gpu if torch.cuda.device_count() > 1 else False

        gpus = request.get("gpus", "all")
        gpus = list(range(torch.cuda.device_count())) if gpus == "all" else [int(g) for g in gpus.split(",")]
        multi_gpu = True if force_multi_gpu or multi_gpu and len(gpus) > 1 else False
        logger.info(f"Using Multi GPU: {multi_gpu}; GPUS: {gpus}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        device = name_to_device(request.get("device", "cuda"))
        logger.info(f"Using device: {device}; Type: {type(device)}")

        tracking = request.get(
            "tracking", "mlflow" if self.enable_tracking and settings.MONAI_LABEL_TRACKING_ENABLED else ""
        )
        tracking = tracking[0] if isinstance(tracking, list) else tracking
        tracking_uri = request.get("tracking_uri")
        tracking_uri = tracking_uri if tracking_uri else settings.MONAI_LABEL_TRACKING_URI
        tracking_experiment_name = request.get("tracking_experiment_name")
        tracking_experiment_name = tracking_experiment_name if tracking_experiment_name else request.get("model")
        tracking_run_name = request.get("tracking_run_name")
        logger.info(f"(Experiment Management) Tracking: {tracking}")
        logger.info(f"(Experiment Management) Tracking URI: {tracking_uri}")
        logger.info(f"(Experiment Management) Experiment Name: {tracking_experiment_name}")
        logger.info(f"(Experiment Management) Run Name: {tracking_run_name}")

        train_handlers = self.bundle_config.get(self.const.key_train_handlers(), [])

        model_filename = request.get("model_filename", "model.pt")
        model_filename = model_filename if isinstance(model_filename, str) else model_filename[0]
        model_pytorch = os.path.join(self.bundle_path, "models", model_filename)

        self._load_checkpoint(model_pytorch, pretrained, train_handlers)

        overrides = {
            self.const.key_bundle_root(): self.bundle_path,
            self.const.key_train_trainer_max_epochs(): max_epochs,
            self.const.key_train_dataset_data(): train_ds,
            self.const.key_device(): device,
            self.const.key_train_handlers(): train_handlers,
        }

        # update config options from user
        for k in self.const.key_displayable_configs():
            if self.bundle_config.get(k):
                displayable_configs = self.bundle_config.get_parsed_content(k, instantiate=True)
                overrides[k] = {c: request[c] for c in displayable_configs.keys()}

        if tracking and tracking.lower() != "none":
            overrides[self.const.key_tracking()] = tracking
            if tracking_uri:
                overrides[self.const.key_tracking_uri()] = tracking_uri
            if tracking_experiment_name:
                overrides[self.const.key_experiment_name()] = tracking_experiment_name
            if tracking_run_name:
                overrides[self.const.key_run_name()] = tracking_run_name

        # external validation datalist supported through bundle itself (pass -1 in the request to use the same)
        if val_ds is not None:
            overrides[self.const.key_validate_dataset_data()] = val_ds

        # allow derived class to update further overrides
        self._update_overrides(overrides)

        if multi_gpu:
            config_paths = [
                c
                for c in self.const.multi_gpu_configs()
                if os.path.exists(os.path.join(self.bundle_path, "configs", c))
            ]
            if not config_paths:
                logger.warning(
                    f"Ignore Multi-GPU Training; No multi-gpu train config {self.const.multi_gpu_configs()} exists"
                )
                return

            train_path = os.path.join(self.bundle_path, "configs", "monailabel_train.json")
            multi_gpu_train_path = os.path.join(self.bundle_path, "configs", config_paths[0])
            logging_file = os.path.join(self.bundle_path, "configs", "logging.conf")
            for k, v in overrides.items():
                self.bundle_config.set(v, k)
            ConfigParser.export_config_file(self.bundle_config.config, train_path, indent=2)  # type: ignore

            sys.path.insert(0, self.bundle_path)
            unload_module("scripts")

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
                run_id,  # run_id, user can pass the arg
                "--meta_file",
                self.bundle_metadata_path,
                "--config_file",
                f"['{train_path}','{multi_gpu_train_path}']",
                "--logging_file",
                logging_file,
            ]

            if tracking:
                cmd.extend(["--tracking", tracking])
                if tracking_uri:
                    cmd.extend(["--tracking_uri", tracking_uri])

            self.run_multi_gpu(request, cmd, env)
        else:
            sys.path.insert(0, self.bundle_path)
            unload_module("scripts")

            self.run_single_gpu(request, overrides)

        sys.path.remove(self.bundle_path)

        logger.info("Training Finished....")
        return {}

    def run_single_gpu(self, request, overrides):
        run_id = request.get("run_id", "run")
        monai.bundle.run(
            run_id=run_id,
            init_id=None,
            final_id=None,
            meta_file=self.bundle_metadata_path,
            config_file=self.bundle_config_path,
            **overrides,
        )

    def run_multi_gpu(self, request, cmd, env):
        self._run_command(cmd, env)

    def _run_command(self, cmd, env):
        logger.info(f"RUNNING COMMAND:: {cmd}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, env=env)
        while process.poll() is None:
            line = process.stdout.readline()
            line = line.rstrip()
            if line:
                print(line, flush=True)

        logger.info(f"Return code: {process.returncode}")
        process.stdout.close()

    def _load_bundle_config(self, path, config):
        bundle_config = ConfigParser()
        bundle_config.read_config(config)
        bundle_config.config.update({self.const.key_bundle_root(): path})  # type: ignore
        return bundle_config

    def _update_overrides(self, overrides):
        return overrides
