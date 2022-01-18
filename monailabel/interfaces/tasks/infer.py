# Copyright 2020 - 2021 MONAI Consortium
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
from abc import abstractmethod
from typing import Dict

import torch

from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.interfaces.utils.transform import run_transforms
from monailabel.transform.writer import Writer

logger = logging.getLogger(__name__)


class InferType:
    """
    Type of Inference Model

    Attributes:
        SEGMENTATION -            Segmentation Model
        ANNOTATION -              Annotation Model
        CLASSIFICATION -          Classification Model
        DEEPGROW -                Deepgrow Interactive Model
        DEEPEDIT -                DeepEdit Interactive Model
        SCRIBBLES -               Scribbles Model
        OTHERS -                  Other Model Type
    """

    SEGMENTATION = "segmentation"
    ANNOTATION = "annotation"
    CLASSIFICATION = "classification"
    DEEPGROW = "deepgrow"
    DEEPEDIT = "deepedit"
    SCRIBBLES = "scribbles"
    OTHERS = "others"
    KNOWN_TYPES = [SEGMENTATION, ANNOTATION, CLASSIFICATION, DEEPGROW, DEEPEDIT, SCRIBBLES, OTHERS]


class InferTask:
    """
    Basic Inference Task Helper
    """

    def __init__(
        self,
        path,
        network,
        type: InferType,
        labels,
        dimension,
        description,
        model_state_dict="model",
        input_key="image",
        output_label_key="pred",
        output_json_key="result",
        config=None,
        load_strict=False,
    ):
        """
        :param path: Model File Path. Supports multiple paths to support versions (Last item will be picked as latest)
        :param network: Model Network (e.g. monai.networks.xyz).  None in case if you use TorchScript (torch.jit).
        :param type: Type of Infer (segmentation, deepgrow etc..)
        :param dimension: Input dimension
        :param description: Description
        :param model_state_dict: Key for loading the model state from checkpoint
        :param input_key: Input key for running inference
        :param output_label_key: Output key for storing result/label of inference
        :param output_json_key: Output key for storing result/label of inference
        :param config: K,V pairs to be part of user config
        :param load_strict: Load model in strict mode
        """
        self.path = path
        self.network = network
        self.type = type
        self.labels = [] if labels is None else [labels] if isinstance(labels, str) else labels
        self.dimension = dimension
        self.description = description
        self.model_state_dict = model_state_dict
        self.input_key = input_key
        self.output_label_key = output_label_key
        self.output_json_key = output_json_key
        self.load_strict = load_strict

        self._networks: Dict = {}
        self._config = {
            # "device": "cuda",
            # "result_extension": None,
            # "result_dtype": None,
            # "result_compress": False
        }
        if config:
            self._config.update(config)

    def info(self):
        return {
            "type": self.type,
            "labels": self.labels,
            "dimension": self.dimension,
            "description": self.description,
            "config": self.config(),
        }

    def config(self):
        return self._config

    def is_valid(self):
        if self.network or self.type == InferType.SCRIBBLES:
            return True

        paths = [self.path] if isinstance(self.path, str) else self.path
        for path in reversed(paths):
            if os.path.exists(path):
                return True
        return False

    def get_path(self):
        if not self.path:
            return None

        paths = [self.path] if isinstance(self.path, str) else self.path
        for path in reversed(paths):
            if os.path.exists(path):
                return path
        return None

    @abstractmethod
    def pre_transforms(self):
        """
        Provide List of pre-transforms

            For Example::

                return [
                    monai.transforms.LoadImaged(keys='image'),
                    monai.transforms.AddChanneld(keys='image'),
                    monai.transforms.Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0]),
                    monai.transforms.ScaleIntensityRanged(keys='image',
                        a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                ]

        """
        pass

    def inverse_transforms(self):
        """
        Provide List of inverse-transforms.  They are normally subset of pre-transforms.
        This task is performed on output_label (using the references from input_key)

        Return one of the following.
            - None: Return None to disable running any inverse transforms (default behavior).
            - Empty: Return [] to run all applicable pre-transforms which has inverse method
            - list: Return list of specific pre-transforms names/classes to run inverse method

            For Example::

                return [
                    monai.transforms.Spacingd,
                ]

        """
        return None

    @abstractmethod
    def post_transforms(self):
        """
        Provide List of post-transforms

            For Example::

                return [
                    monai.transforms.AddChanneld(keys='pred'),
                    monai.transforms.Activationsd(keys='pred', softmax=True),
                    monai.transforms.AsDiscreted(keys='pred', argmax=True),
                    monai.transforms.SqueezeDimd(keys='pred', dim=0),
                    monai.transforms.ToNumpyd(keys='pred'),
                    monailabel.interface.utils.Restored(keys='pred', ref_image='image'),
                    monailabel.interface.utils.ExtremePointsd(keys='pred', result='result', points='points'),
                    monailabel.interface.utils.BoundingBoxd(keys='pred', result='result', bbox='bbox'),
                ]

        """
        pass

    @abstractmethod
    def inferer(self):
        """
        Provide Inferer Class

            For Example::

                return monai.inferers.SlidingWindowInferer(roi_size=[160, 160, 160])
        """
        pass

    def __call__(self, request):
        """
        It provides basic implementation to run the following in order
            - Run Pre Transforms
            - Run Inferer
            - Run Post Transforms
            - Run Writer to save the label mask and result params

        Returns: Label (File Path) and Result Params (JSON)
        """
        begin = time.time()
        req = copy.deepcopy(self._config)
        req.update(copy.deepcopy(request))
        logger.info(f"Infer Request (final): {req}")

        data = copy.deepcopy(req)
        data.update({"image_path": req.get("image")})
        device = req.get("device", "cuda")

        start = time.time()
        pre_transforms = self.pre_transforms()
        data = self.run_pre_transforms(data, pre_transforms)
        latency_pre = time.time() - start

        start = time.time()
        data = self.run_inferer(data, device=device)
        latency_inferer = time.time() - start

        start = time.time()
        data = self.run_invert_transforms(data, pre_transforms, self.inverse_transforms())
        latency_invert = time.time() - start

        start = time.time()
        data = self.run_post_transforms(data, self.post_transforms())
        latency_post = time.time() - start

        start = time.time()
        result_file_name, result_json = self.writer(data)
        latency_write = time.time() - start

        latency_total = time.time() - begin
        logger.info(
            "++ Latencies => Total: {:.4f}; "
            "Pre: {:.4f}; Inferer: {:.4f}; Invert: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total,
                latency_pre,
                latency_inferer,
                latency_invert,
                latency_post,
                latency_write,
            )
        )

        result_json["label_names"] = self.labels

        logger.info("Result File: {}".format(result_file_name))
        logger.info("Result Json: {}".format(result_json))
        return result_file_name, result_json

    def run_pre_transforms(self, data, transforms):
        return run_transforms(data, transforms, log_prefix="PRE", use_compose=False)

    def run_invert_transforms(self, data, pre_transforms, names):
        if names is None:
            return data

        pre_names = dict()
        transforms = []
        for t in reversed(pre_transforms):
            if hasattr(t, "inverse"):
                pre_names[t.__class__.__name__] = t
                transforms.append(t)

        # Run only selected/given
        if len(names) > 0:
            transforms = [pre_transforms[n if isinstance(n, str) else n.__name__] for n in names]

        d = copy.deepcopy(dict(data))
        d[self.input_key] = data[self.output_label_key]

        d = run_transforms(d, transforms, inverse=True, log_prefix="INV")
        data[self.output_label_key] = d[self.input_key]
        return data

    def run_post_transforms(self, data, transforms):
        return run_transforms(data, transforms, log_prefix="POST")

    def _get_network(self, device):
        path = self.get_path()
        logger.info("Infer model path: {}".format(path))
        if not path and not self.network:
            if self.type == InferType.SCRIBBLES:
                return None

            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                f"Model Path ({self.path}) does not exist/valid",
            )

        cached = self._networks.get(device)
        statbuf = os.stat(path) if path else None
        network = None
        if cached:
            if statbuf and statbuf.st_mtime == cached[1]:
                network = cached[0]
            elif statbuf:
                logger.info(f"Reload model from cache.  Prev ts: {cached[1]}; Current ts: {statbuf.st_mtime}")

        if network is None:
            if self.network:
                network = self.network
                if path:
                    # If we are using a CPU-only machine, try to load the network for CPU inference
                    if torch.cuda.is_available():
                        checkpoint = torch.load(path)
                    else:
                        checkpoint = torch.load(path, map_location=torch.device("cpu"))

                    model_state_dict = checkpoint.get(self.model_state_dict, checkpoint)
                    network.load_state_dict(model_state_dict, strict=self.load_strict)
            else:
                # If we are using a CPU-only machine, try to load the network for CPU inference
                if torch.cuda.is_available():
                    network = torch.jit.load(path)
                else:
                    network = torch.jit.load(path, map_location=torch.device("cpu"))

            if device == "cuda":
                network = network.cuda()

            network.eval()
            self._networks[device] = (network, statbuf.st_mtime if statbuf else 0)

        return network

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """

        inferer = self.inferer()
        logger.info("Running Inferer:: {}".format(inferer.__class__.__name__))

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        network = self._get_network(device)
        if network:
            inputs = data[self.input_key]
            inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
            inputs = inputs[None] if convert_to_batch else inputs
            if device == "cuda":
                inputs = inputs.cuda()

            with torch.no_grad():
                outputs = inferer(inputs, network)
            if device == "cuda":
                torch.cuda.empty_cache()

            outputs = outputs[0] if convert_to_batch else outputs
            data[self.output_label_key] = outputs
        else:
            # consider them as callable transforms
            data = run_transforms(data, inferer, log_prefix="INF", log_name="Inferer")
        return data

    def writer(self, data, extension=None, dtype=None):
        """
        You can provide your own writer.  However this writer saves the prediction/label mask to file
        and fetches result json

        :param data: typically it is post processed data
        :param extension: output label extension
        :param dtype: output label dtype
        :return: tuple of output_file and result_json
        """
        logger.info("Writing Result")
        if extension is not None:
            data["result_extension"] = extension
        if dtype is not None:
            data["result_dtype"] = dtype

        writer = Writer(label=self.output_label_key, json=self.output_json_key)
        return writer(data)

    def clear(self):
        self._networks.clear()
