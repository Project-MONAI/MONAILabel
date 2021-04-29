import copy
import logging
import os
import time
from abc import abstractmethod

import torch

from monailabel.interface.exception import MONAILabelError, MONAILabelException
from monailabel.utils.others.writer import Writer

logger = logging.getLogger(__name__)


class InferType:
    SEGMENTATION = 'segmentation'
    CLASSIFICATION = 'classification'
    DEEPGROW = 'deepgrow'
    DEEPEDIT = "deepedit"
    OTHERS = 'others'
    KNOWN_TYPES = [SEGMENTATION, CLASSIFICATION, DEEPGROW, OTHERS]


class InferenceTask:
    """
    Basic Inference Task Helper
    """

    def __init__(self, path, network, type: InferType, labels, dimension, description):
        """
        :param path: Model File Path
        :param network: Model Network (e.g. monai.networks.xyz).  None in case if you use TorchScript (torch.jit).
        :param type: Type of Infer (segmentation, deepgrow etc..)
        :param dimension: Input dimension
        :param description: Description
        """
        self.path = path
        self.network = network
        self.type = type
        self.labels = labels
        self.dimension = dimension
        self.description = description

        self._networks = {}

    def info(self):
        return {
            "type": self.type,
            "labels": self.labels,
            "dimension": self.dimension,
            "description": self.description,
        }

    def is_valid(self):
        return os.path.exists(os.path.join(self.path))

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
        data.update({'image_path': request.get('image')})
        device = request.get('device', 'cuda')

        start = time.time()
        data = self.run_pre_transforms(data, self.pre_transforms())
        latency_pre = time.time() - start

        start = time.time()
        data = self.run_inferer(data, device=device)
        latency_inferer = time.time() - start

        start = time.time()
        data = self.run_post_transforms(data, self.post_transforms())
        latency_post = time.time() - start

        start = time.time()
        result_file_name, result_json = self.writer(data)
        latency_write = time.time() - start

        latency_total = time.time() - begin
        logger.info(
            "++ Latencies => Total: {:.4f}; Pre: {:.4f}; Inferer: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total, latency_pre, latency_inferer, latency_post, latency_write))

        logger.info('Result File: {}'.format(result_file_name))
        logger.info('Result Json: {}'.format(result_json))
        return result_file_name, result_json

    def run_pre_transforms(self, data, transforms):
        return self.run_transforms(data, transforms, log_prefix='PRE')

    def run_post_transforms(self, data, transforms):
        return self.run_transforms(data, transforms, log_prefix='POST')

    def run_inferer(self, data, convert_to_batch=True, device='cuda', input_key='image', output_key='pred'):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :param input_key: input key in data to feed it to inferer
        :param output_key: output key to store the predictions
        :return: updated data with output_key stored that will be used for post-processing
        """
        if not os.path.exists(os.path.join(self.path)):
            raise MONAILabelException(MONAILabelError.INFERENCE_ERROR, f"Model Path ({self.path}) does not exist")

        inputs = data[input_key]
        inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
        inputs = inputs[None] if convert_to_batch else inputs
        inputs = inputs.cuda() if device == 'cuda' else inputs

        network = self._networks.get(device)
        if network is None:
            if self.network:
                network = self.network
                network.load_state_dict(torch.load(self.path))
            else:
                network = torch.jit.load(self.path)

            network = network.cuda() if device == 'cuda' else network
            network.eval()
            self._networks[device] = network

        inferer = self.inferer()
        logger.info("Running Inferer:: {}".format(inferer.__class__.__name__))

        with torch.no_grad():
            outputs = inferer(inputs, network)
        if device == 'cuda':
            torch.cuda.empty_cache()

        outputs = outputs[0] if convert_to_batch else outputs
        data[output_key] = outputs
        return data

    def writer(self, data, label='pred', text='result', extension=None, dtype=None):
        """
        You can provide your own writer.  However this writer saves the prediction/label mask to file
        and fetches result json

        :param data: typically it is post processed data
        :param label: label that needs to be written
        :param text: text field from data which represents result params
        :param extension: output label extension
        :param dtype: output label dtype
        :return: tuple of output_file and result_json
        """
        logger.info('Writing Result')
        if extension is not None:
            data['result_extension'] = extension
        if dtype is not None:
            data['result_dtype'] = dtype

        writer = Writer(label=label, json=text)
        return writer(data)

    @staticmethod
    def dump_data(data):
        if logging.getLogger().level == logging.DEBUG:
            logger.debug('**************************** DATA ********************************************')
            for k in data:
                v = data[k]
                logger.debug('Data key: {} = {}'.format(
                    k,
                    v.shape if hasattr(v, 'shape') else v if type(v) in (
                        int, float, bool, str, dict, tuple, list) else type(v)))
            logger.debug('******************************************************************************')

    @staticmethod
    def _shape_info(data, keys=('image', 'label', 'pred', 'model')):
        shape_info = []
        for key in keys:
            val = data.get(key)
            if val is not None and hasattr(val, 'shape'):
                shape_info.append('{}: {}'.format(key, val.shape))
        return '; '.join(shape_info)

    @staticmethod
    def run_transforms(data, transforms, log_prefix='POST'):
        """
        Run Transforms

        :param data: Input data dictionary
        :param transforms: List of transforms to run
        :param log_prefix: Logging prefix (POST or PRE)
        :return: Processed data after running transforms
        """
        logger.info('{} - Run Transforms'.format(log_prefix))
        logger.info('{} - Input Keys: {}'.format(log_prefix, data.keys()))

        if not transforms:
            return data

        for t in transforms:
            name = t.__class__.__name__
            start = time.time()

            InferenceTask.dump_data(data)
            if callable(t):
                data = t(data)
            else:
                raise MONAILabelException(MONAILabelError.INFERENCE_ERROR, "Transformer '{}' is not callable".format(
                    t.__class__.__name__))

            logger.info("{} - Transform ({}): Time: {:.4f}; {}".format(
                log_prefix, name, float(time.time() - start), InferenceTask._shape_info(data)))
            logger.debug('-----------------------------------------------------------------------------')

        InferenceTask.dump_data(data)
        return data
