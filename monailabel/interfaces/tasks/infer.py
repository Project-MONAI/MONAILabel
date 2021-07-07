import copy
import logging
import os
import time
from abc import abstractmethod
from typing import Dict

import torch

from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.utils.others.writer import Writer

logger = logging.getLogger(__name__)


class InferType:
    """
    Type of Inference Model
    Attributes:
        SEGMENTATION -            Segmentation Model
        CLASSIFICATION -          Classification Model
        DEEPGROW -                Deepgrow Interactive Model
        DEEPEDIT -                DeepEdit Interactive Model
        POSTPROCS -               Post-Processor Model
        OTHERS -                  Other Model Type
    """

    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    DEEPGROW = "deepgrow"
    DEEPEDIT = "deepedit"
    POSTPROCS = "postprocessor"
    OTHERS = "others"
    KNOWN_TYPES = [SEGMENTATION, CLASSIFICATION, DEEPGROW, POSTPROCS, OTHERS]


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

        self._networks: Dict = {}

    def info(self):
        return {
            "type": self.type,
            "labels": self.labels,
            "dimension": self.dimension,
            "description": self.description,
        }

    def is_valid(self):
        if self.network or self.type == InferType.POSTPROCS:
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

        data = copy.deepcopy(request)
        data.update({"image_path": request.get("image")})
        device = request.get("device", "cuda")

        start = time.time()
        pre_transforms = self.pre_transforms()
        data = self.run_pre_transforms(data, pre_transforms)
        latency_pre = time.time() - start

        start = time.time()
        if self.type == InferType.POSTPROCS:
            data = self.run_postprocessor(data)
        else:
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

        logger.info("Result File: {}".format(result_file_name))
        logger.info("Result Json: {}".format(result_json))
        return result_file_name, result_json

    def run_pre_transforms(self, data, transforms):
        return self.run_transforms(data, transforms, log_prefix="PRE")

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

        d = self.run_transforms(d, transforms, inverse=True, log_prefix="INV")
        data[self.output_label_key] = d[self.input_key]
        return data

    def run_post_transforms(self, data, transforms):
        return self.run_transforms(data, transforms, log_prefix="POST")

    def run_postprocessor(self, data):
        p = self.inferer()
        logger.info("Running PostProcessor: {}".format(p))
        return p(data) if p else data

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """
        path = self.get_path()
        logger.info("Infer model path: {}".format(path))
        if not path and not self.network:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                f"Model Path ({self.path}) does not exist/valid",
            )

        inputs = data[self.input_key]
        inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
        inputs = inputs[None] if convert_to_batch else inputs
        inputs = inputs.cuda() if device == "cuda" else inputs

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
                    checkpoint = torch.load(path)
                    model_state_dict = checkpoint.get(self.model_state_dict, checkpoint)
                    network.load_state_dict(model_state_dict)
            else:
                network = torch.jit.load(path)

            network = network.cuda() if device == "cuda" else network
            network.eval()
            self._networks[device] = (network, statbuf.st_mtime if statbuf else 0)

        inferer = self.inferer()
        logger.info("Running Inferer:: {}".format(inferer.__class__.__name__))

        with torch.no_grad():
            outputs = inferer(inputs, network)
        if device == "cuda":
            torch.cuda.empty_cache()

        outputs = outputs[0] if convert_to_batch else outputs
        data[self.output_label_key] = outputs
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

    @staticmethod
    def dump_data(data):
        if logging.getLogger().level == logging.DEBUG:
            logger.debug("**************************** DATA ********************************************")
            for k in data:
                v = data[k]
                logger.debug(
                    "Data key: {} = {}".format(
                        k,
                        v.shape
                        if hasattr(v, "shape")
                        else v
                        if type(v) in (int, float, bool, str, dict, tuple, list)
                        else type(v),
                    )
                )
            logger.debug("******************************************************************************")

    @staticmethod
    def _shape_info(data, keys=("image", "label", "pred", "model")):
        shape_info = []
        for key in keys:
            val = data.get(key)
            if val is not None and hasattr(val, "shape"):
                shape_info.append("{}: {}".format(key, val.shape))
        return "; ".join(shape_info)

    @staticmethod
    def run_transforms(data, transforms, inverse=False, log_prefix="POST"):
        """
        Run Transforms

        :param data: Input data dictionary
        :param transforms: List of transforms to run
        :param inverse: Run inverse instead of call/forward function
        :param log_prefix: Logging prefix (POST or PRE)
        :return: Processed data after running transforms
        """
        logger.info("{} - Run Transforms".format(log_prefix))
        logger.info("{} - Input Keys: {}".format(log_prefix, data.keys()))

        if not transforms:
            return data

        for t in transforms:
            name = t.__class__.__name__
            start = time.time()

            InferTask.dump_data(data)
            if inverse:
                if hasattr(t, "inverse"):
                    data = t.inverse(data)
                else:
                    raise MONAILabelException(
                        MONAILabelError.INFERENCE_ERROR,
                        "Transformer '{}' has no invert method".format(t.__class__.__name__),
                    )
            elif callable(t):
                data = t(data)
            else:
                raise MONAILabelException(
                    MONAILabelError.INFERENCE_ERROR,
                    "Transformer '{}' is not callable".format(t.__class__.__name__),
                )

            logger.info(
                "{} - Transform ({}): Time: {:.4f}; {}".format(
                    log_prefix,
                    name,
                    float(time.time() - start),
                    InferTask._shape_info(data),
                )
            )
            logger.debug("-----------------------------------------------------------------------------")

        InferTask.dump_data(data)
        return data
