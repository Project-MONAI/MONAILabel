import copy
import time
import logging

from lib.transforms import ConvertLogitsToBinaryd
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CopyItemsd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    SqueezeDimd,
    ToNumpyd,
)
from monailabel.utils.infer import InferenceTask, InferType
from monailabel.utils.others.post import Restored, BoundingBoxd

logger = logging.getLogger(__name__)

class MyInfer(InferenceTask):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.SEGMENTATION,
            labels=["lesion"],
            dimension=3,
            description='A pre-trained model for volumetric (3D) segmentation of the spleen from CT image'
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys='image'),
            AddChanneld(keys='image'),
            # Spacingd(keys='image', pixdim=[1.0, 1.0, 1.0]),
            # ScaleIntensityRanged(keys='image', a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=[160, 160, 160])

    def post_transforms(self):
        return [
            AddChanneld(keys='pred'),
            Activationsd(keys='pred', softmax=True),
            ConvertLogitsToBinaryd(key='pred', foreground_class=[1], softmax=False),
            CopyItemsd(keys='pred', times=1, names='logits'),
            AsDiscreted(keys='pred', argmax=True),
            SqueezeDimd(keys=['pred', 'logits'], dim=0),
            ToNumpyd(keys=['pred', 'logits']),
            Restored(keys=['pred', 'logits'], ref_image='image'),
            BoundingBoxd(keys='pred', result='result', bbox='bbox'),
        ]
    
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
        # save logits file temporarily somewhere
        logits_file_name, logits_json = self.writer(data, label='logits')
        latency_write = time.time() - start
        
        latency_total = time.time() - begin
        logger.info(
            "++ Latencies => Total: {:.4f}; Pre: {:.4f}; Inferer: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total, latency_pre, latency_inferer, latency_post, latency_write))
        
        logger.info('Logits File: {}'.format(logits_file_name))
        logger.info('Logits Json: {}'.format(logits_json))

        logger.info('Result File: {}'.format(result_file_name))
        logger.info('Result Json: {}'.format(result_json))

        result_json['logits_file_name'] = logits_file_name
        result_json['logits_json'] = logits_json
        return result_file_name, result_json
    