import tempfile
import logging
import os
import io
import pathlib
import atexit

import torch
import yaml


from lib import MyInfer, MyTrain, MyActiveLearning
from lib.transforms import AddUnaryTermd, ApplyCRFPostProcd
from monai.transforms import (
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    Spacingd,
    Activationsd,
    AsDiscreted,
    ToNumpyd,
    NormalizeIntensityd,
    AsChannelLastd,
    SqueezeDimd,
    ScaleIntensityRanged,
    ToTensord,
    SaveImaged,
)
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
        scribbles = io.BytesIO(open(request['scribbles'], 'rb').read())

        scribbles_file = tempfile.NamedTemporaryFile(suffix=".nii.gz").name
        with open(scribbles_file, 'wb') as f:
            scribbles.seek(0)
            f.write(scribbles.getbuffer())       
        img_name = os.path.basename(request['image']).rsplit('.')[0]
        
        data['scribbles'] = scribbles_file
        data['logits'] = self.logits_files[img_name]['file']
        data['image'] = request['image']

        print('scribbles: {}\nlogits: {}\nimg_name: {}\n\n'.format(data['scribbles'], data['logits'], data['image']))

        def apply_tx(_data, tx):
            to_print = ''
            print('Applying {}'.format(tx.__class__.__name__))
            print('Before tx:')
            for key in sorted(_data.keys()):
                if hasattr(_data[key], 'shape'):
                    to_print += '{}: {} | '.format(key, _data[key].shape)
            print(to_print)
            _data = tx(_data)
            to_print = ''
            print('After tx:')
            for key in sorted(_data.keys()):
                if hasattr(_data[key], 'shape'):
                    to_print += '{}: {} | '.format(key, _data[key].shape)
            print(to_print)
            print()
            return _data
        use_simplecrf = False
        pre_transforms = [
            LoadImaged(keys=['image', 'logits', 'scribbles']),
            AddChanneld(keys=['image', 'logits', 'scribbles']),
            Spacingd(keys=['image', 'logits'], pixdim=[3.0, 3.0, 5.0]),
            Spacingd(keys=['scribbles'], pixdim=[3.0, 3.0, 5.0], mode='nearest'),
            ScaleIntensityRanged(keys='image', a_min=-164, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            AddUnaryTermd(ref_prob='logits', unary="unary", scribbles="scribbles", channel_dim=0, sc_background_label=2, sc_foreground_label=3, scale_infty=10, use_simplecrf=use_simplecrf),
            AddChanneld(keys=['image', 'unary']),
            ToTensord(keys=['image', 'logits', 'unary'])

        ]

        post_transforms = [
            ApplyCRFPostProcd(unary='unary', pairwise='image', post_proc_label='pred', device=torch.device('cpu'), use_simplecrf=use_simplecrf),
            # AddChanneld(keys='pred'),
            # CopyItemsd(keys='pred', times=1, names='logits'),
            # Activationsd(keys='pred', softmax=True),
            # AsDiscreted(keys='pred', argmax=True),
            SqueezeDimd(keys=['pred', 'logits'], dim=0),
            ToNumpyd(keys=['pred', 'logits']),
            Restored(keys='pred', ref_image='image'),
            # BoundingBoxd(keys='pred', result='result', bbox='bbox'),
            # SaveImaged(keys='pred', output_dir='/tmp', output_postfix='postproc', output_ext='.nii.gz', resample=False),
        ]

        for prtx in pre_transforms:
            data = apply_tx(data, prtx)
        # import numpy as np
        # print(np.unique(data['scribbles']))
        for potx in post_transforms:
            data = apply_tx(data, potx)

        # os.unlink(scribbles_file)
        print(data['logits_meta_dict']['affine'])
        print(data['scribbles_meta_dict']['affine'])
        print(data['image_meta_dict']['affine'])
        import nibabel as nib
        result_img_file = '/tmp/postproclabel.nii.gz'
        results_img = nib.Nifti1Image(data['pred'], data['image_meta_dict']['affine'])
        nib.save(results_img, '/tmp/postproclabel.nii.gz')

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
        return {"label": result_img_file, "params": {}}
