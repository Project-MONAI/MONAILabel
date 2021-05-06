import tempfile
import logging
import os
import io
import pathlib
import atexit

import yaml


from lib import MyInfer, MyTrain, MyActiveLearning
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monailabel.interfaces.app import MONAILabelApp
from monailabel.utils.infer.deepgrow_2d import InferDeepgrow2D
from monailabel.utils.infer.deepgrow_3d import InferDeepgrow3D
from monailabel.interfaces.exception import MONAILabelException, MONAILabelError

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        model_dir = os.path.join(app_dir, "model")
        infers = {
            "deepgrow_2d": InferDeepgrow2D(os.path.join(model_dir, "deepgrow_2d.ts")),
            "deepgrow_3d": InferDeepgrow3D(os.path.join(model_dir, "deepgrow_3d.ts")),
            "segmentation_spleen": MyInfer(os.path.join(model_dir, "segmentation_spleen.ts")),
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
            print('We are HERE!')
            print(self.logits_files)

        return {"label": result_file_name, "params": result_json}

    def cleanup_logits_files(self):
        # clean residual logits help from: https://stackoverflow.com/a/32732654
        for key in list(self.logits_files.keys()):
            if key in self.logits_files.keys():
                print('removing temp logits file for: {}'.format(key))
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
    
    def postproc_label(self, request):
        """
        Saving New Label.  You can extend this has callback handler to run calibrations etc. over Active learning models

        Args:
            request: JSON object which contains Label and Image details

                For example::

                    {
                        "image": "file://xyz.com",
                        "label": "file://label_xyz.com",
                        "segments" ["spleen"],
                        "params": {},
                    }

        Returns:
            JSON containing next image and label info
        """
        print('*'*100)
        # prepare data dictonary
        data = {}
        # TODO: rename label to scribbles
        label = io.BytesIO(open(request['label'], 'rb').read())

        scribbles_in = tempfile.NamedTemporaryFile(suffix=".nii.gz").name
        with open(scribbles_in, 'wb') as f:
            label.seek(0)
            f.write(label.getbuffer())       
        img_name = os.path.basename(request['image']).rsplit('.')[0]
        
        data['scribbles'] = scribbles_in
        data['logits'] = self.logits_files[img_name]['file']
        data['image'] = request['image']

        print('scribbles: {}\nlogits: {}\n img_name: {}\n\n'.format(data['scribbles'], data['logits'], data['image']))

        # os.unlink(scribbles_in)

        # file_ext = ''.join(pathlib.Path(request['label']).suffixes)
        # segments = request.get('segments')
        # if not segments:
        #     segments = self.info().get("labels", [])
        # segments = [segments] if isinstance(segments, str) else segments
        # segments = "+".join(segments) if len(segments) else "unk"

        # label_id = f"label_{segments}_{img_name}{file_ext}"
        # # label_file = self.datastore().save_label(request['image'], label_id, label)

        

        # return {
        #     "image": request.get("image"),
        #     "label": label_file,
        # }

        return {"label": scribbles_in, "params": {'bbox': [[167, 10], [357, 83]]}}
