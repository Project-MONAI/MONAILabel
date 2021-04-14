import copy
import logging
import time
from abc import abstractmethod

import torch

from monailabel.interface.exception import MONAILabelError, MONAILabelException
from monailabel.interface.utils.writer import Writer

logger = logging.getLogger(__name__)


class InferenceEngine(object):
    """
    Basic Inference Engine Helper
    """

    def __init__(self, model):
        """
        :param model: Pre-Trained Model File (TorchScript)
        """
        self._model = model
        self._networks = {}

    @abstractmethod
    def pre_transforms(self):
        """
        Provide List of pre-transforms

            For Example::

                return [
                    monai.transforms.LoadImaged(keys='image'),
                    monai.transforms.AddChanneld(keys='image'),
                    monai.transforms.Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0]),
                    monai.transforms.ScaleIntensityRanged(keys='image', a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
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

    def run(self, data_file, params, device):
        """
        It provides basic implementation to run the following in order
            - Run Pre Transforms
            - Run Inferer
            - Run Post Transforms
            - Run Writer to save the label mask and result params

        Returns: Label (File Path) and Result Params (JSON)
        """
        begin = time.time()
        data = copy.deepcopy(params)
        data.update({'image': data_file, 'image_path': data_file, 'params': params})

        start = time.time()
        data = self.run_transforms(data, self.pre_transforms(), log_prefix='PRE')
        latency_pre = time.time() - start

        start = time.time()
        data = self.run_inferer(data, device=device)
        latency_inferer = time.time() - start

        start = time.time()
        data = self.run_transforms(data, self.post_transforms(), log_prefix='POST')
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
        logger.info('Running Inferer')
        inputs = data[input_key]
        inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
        inputs = inputs[None] if convert_to_batch else inputs
        inputs = inputs.cuda() if device == 'cuda' else inputs

        network = self._networks.get(device)
        if network is None:
            network = torch.jit.load(self._model)
            network = network.cuda() if device == 'cuda' else network
            network.eval()
            self._networks[device] = network

        inferer = self.inferer()
        logger.info("Inferer:: {}".format(inferer.__class__.__name__))
        with torch.no_grad():
            outputs = inferer(inputs, network)
        if device == 'cuda':
            torch.cuda.empty_cache()

        outputs = outputs[0] if convert_to_batch else outputs
        data[output_key] = outputs
        return data

    def writer(self, data, image='pred', text='result', image_ext=None, image_dtype=None):
        """
        You can provide your own writer.  However this writer saves the prediction/label mask to file
        and fetches result json

        :param data: typically it is post processed data
        :param image: image that needs to be written
        :param text: text field from data which represents result params
        :param image_ext: output image extension
        :param image_dtype: output image dtype
        :return: tuple of output_file and result_json
        """
        logger.info('Writing Result')
        if image_ext is not None:
            data['result_extension'] = image_ext
        if image_dtype is not None:
            data['result_dtype'] = image_dtype

        writer = Writer(image=image, json=text)
        return writer(data)

    def dump_data(self, data):
        if logging.getLogger().level == logging.DEBUG:
            logger.debug('**************************** DATA ********************************************')
            for k in data:
                v = data[k]
                logger.debug('Data key: {} = {}'.format(
                    k,
                    v.shape if hasattr(v, 'shape') else v if type(v) in (
                        int, float, bool, str, dict, tuple, list) else type(v)))
            logger.debug('******************************************************************************')

    def _shape_info(self, data, keys=('image', 'label', 'pred', 'model')):
        shape_info = []
        for key in keys:
            val = data.get(key)
            if val is not None and hasattr(val, 'shape'):
                shape_info.append('{}: {}'.format(key, val.shape))
        return '; '.join(shape_info)

    def run_transforms(self, data, transforms, log_prefix='POST'):
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

            self.dump_data(data)
            if callable(t):
                data = t(data)
            else:
                raise MONAILabelException(MONAILabelError.INFERENCE_ERROR, "Transformer '{}' is not callable".format(
                    t.__class__.__name__))

            logger.info("{} - Transform ({}): Time: {:.4f}; {}".format(
                log_prefix, name, float(time.time() - start), self._shape_info(data)))
            logger.debug('-----------------------------------------------------------------------------')

        self.dump_data(data)
        return data
