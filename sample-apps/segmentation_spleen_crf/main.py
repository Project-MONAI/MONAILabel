from monailabel.utils import postproc
import tempfile
import logging
import os
import io
import pathlib
import atexit

import torch
import yaml


from lib import MyInfer, MyTrain, MyActiveLearning, MyCRF
from lib.transforms import AddUnaryTermd, ApplyCRFPostProcd
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monailabel.interfaces.app import MONAILabelApp
from monailabel.utils.infer.deepgrow_2d import InferDeepgrow2D
from monailabel.utils.infer.deepgrow_3d import InferDeepgrow3D
from monailabel.interfaces.exception import MONAILabelException, MONAILabelError
from monailabel.utils.others.post import Restored, BoundingBoxd

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        model_dir = os.path.join(app_dir, "model")
        infers = {
            "deepgrow_2d": InferDeepgrow2D(os.path.join(model_dir, "deepgrow_2d.ts")),
            "deepgrow_3d": InferDeepgrow3D(os.path.join(model_dir, "deepgrow_3d.ts")),
            "segmentation_spleen": MyInfer(os.path.join(model_dir, "segmentation_spleen.ts")),
        }

        self.postproc_methods = {
            # can have other post processors here
            "CRF": MyCRF(method='CRF'),
        }

        # define a dictionary to keep track of logits files
        # these are needed for postproc step
        self.logits_files = {}

        # define a cleanup function if application abruptly temrinates, to clean tmp logit files
        atexit.register(self.cleanup_logits_files)

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            active_learning=MyActiveLearning()
        )

    def infer(self, request):
        """
        Run Inference for an exiting pre-trained model.

        Args:
            request: JSON object which contains `model`, `image`, `params` and `device`

                For example::

                    {
                        "device": "cuda"
                        "model": "segmentation_spleen",
                        "image": "file://xyz",
                        "params": {},
                    }

        Raises:
            MONAILabelException: When ``model`` is not found

        Returns:
            JSON containing `label` and `params`
        """
        # do a little cleanup
        self.cleanup_logits_files()
        model_name = request.get('model')
        model_name = model_name if model_name else 'model'

        task = self.infers.get(model_name)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Task is not Initialized. There is no pre-trained model available"
            )

        result_file_name, result_json = task(request)
        if 'logits_file_name' in result_json:
            logits_file = result_json.pop('logits_file_name')
            logits_json = result_json.pop('logits_json')
            self.logits_files[os.path.basename(request.get('image')).rsplit('.')[0]] = {'file': logits_file, 'params': logits_json}
            logger.info(f'Logits files saved: {self.logits_files}')

        return {"label": result_file_name, "params": result_json}

    def cleanup_logits_files(self):
        # clean residual logits help from: https://stackoverflow.com/a/32732654
        for key in list(self.logits_files.keys()):
            if key in self.logits_files.keys():
                logger.info(f'removing temp logits file for: {key}')
                cur_item = self.logits_files.pop(key)
                # del file on disk
                os.unlink(cur_item['file'])
        
    def train(self, request):
        epochs = request.get('epochs', 1)
        amp = request.get('amp', True)
        device = request.get('device', 'cuda')
        lr = request.get('lr', 0.0001)
        val_split = request.get('val_split', 0.2)

        logger.info(f"Training request: {request}")
        task = MyTrain(
            output_dir=os.path.join(self.app_dir, "train", "train_0"),
            data_list=self.datastore().datalist(),
            network=UNet(
                dimensions=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH
            ),
            device=device,
            lr=lr,
            val_split=val_split
        )

        return task(max_epochs=epochs, amp=amp)

    def save_scribbles(self, image_in, scribbles_in):
        scribbles = io.BytesIO(open(scribbles_in, 'rb').read())
        scribbles_file = tempfile.NamedTemporaryFile(suffix=".nii.gz").name
        with open(scribbles_file, 'wb') as f:
            scribbles.seek(0)
            f.write(scribbles.getbuffer())
        return scribbles_file

    def postproc_label(self, request):
        method = request.get('method')
        task = self.postproc_methods.get(method)

        # prepare data
        data = {}
        scribbles_raw = request['scribbles']
        image_name = os.path.basename(request['image']).rsplit('.')[0]
        scribbles_file = self.save_scribbles(image_name, scribbles_raw)

        data['scribbles'] = scribbles_file
        data['logits'] = self.logits_files[image_name]['file']
        data['image'] = request['image']

        logger.info('scribbles: {}\n\tlogits: {}\n\timage_name: {}'.format(data['scribbles'], data['logits'], data['image']))
        result_file_name, result_json = task(data)

        return {'label': result_file_name, 'params': result_json}