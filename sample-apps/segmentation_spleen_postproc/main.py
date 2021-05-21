import atexit
import json
import logging
import os
import pathlib
import time
from shutil import copyfile

from lib import MyInfer, MyStrategy, MyTrain, SpleenCRF
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)

# Whether to save research data or not
RESEARCH_MODE=True

class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.network = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

        self.pretrained_model = os.path.join(self.model_dir, "segmentation_spleen.pt")
        self.final_model = os.path.join(self.model_dir, "final.pt")
        path = [self.pretrained_model, self.final_model]

        infers = {
            "segmentation_spleen": MyInfer(path, self.network),
        }

        postprocs = {
            # can have other post processors here
            "CRF": SpleenCRF(method='CRF'),
        }

        strategies = {
            "random": Random(),
            "first": MyStrategy(),
        }
        resources = [
            (self.pretrained_model, "https://www.dropbox.com/s/xc9wtssba63u7md/segmentation_spleen.pt?dl=1"),
        ]

        # define a dictionary to keep track of logits files
        # these are needed for postproc step
        self.logits_files = {}
        if RESEARCH_MODE:
            self.save_research_sessionid = time.strftime('%Y%m%d%H%M%S') # use current time as unique id of session
            self.save_research_path = os.path.join(studies, 'research_data', self.save_research_sessionid)
            self.save_research_data = os.path.join(self.save_research_path, 'session_data.json')
            
            # create folders/files
            os.makedirs(self.save_research_path, exist_ok=True)
            with open(self.save_research_data, 'w') as fp:
                json.dump({}, fp, indent=4)

            logger.info('Running app in research mode, saving scribbles data to: {}'.format(self.save_research_path))
            

        # define a cleanup function if application abruptly temrinates, to clean tmp logit files
        atexit.register(self.cleanup_logits_files)

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            postprocs=postprocs,
            strategies=strategies,
            resources=resources,
        )

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        self.add_deepgrow_infer_tasks()

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
                        "save_label": "true/false",
                        "label_tag": "my_custom_label_tag", (if not provided defaults to `original`)
                        "params": {},
                    }

        Raises:
            MONAILabelException: When ``model`` is not found

        Returns:
            JSON containing `label` and `params`
        """
        # cleanup previous logits files
        self.cleanup_logits_files()

        model_name = request.get("model")
        model_name = model_name if model_name else "model"

        task = self.infers.get(model_name)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Task is not Initialized. There is no pre-trained model available",
            )

        image_id = request["image"]
        request["image"] = self._datastore.get_image_uri(request["image"])
        result_file_name, result_json = task(request)

        if request.get("save_label", True):
            self.datastore().save_label(
                image_id,
                result_file_name,
                request.get("label_tag") if request.get("label_tag") else DefaultLabelTag.ORIGINAL.value,
            )

        if 'logits_file_name' in result_json:
            logits_file = result_json.pop('logits_file_name')
            logits_json = result_json.pop('logits_json')
            self.logits_files[os.path.basename(request.get('image')).rsplit('.')[0]] = {'file': logits_file, 'params': logits_json}
            logger.info(f'Logits files saved: {self.logits_files}')

        return {"label": result_file_name, "params": result_json}

    def train(self, request):
        name = request.get("name", "model_01")
        epochs = request.get("epochs", 1)
        amp = request.get("amp", True)
        device = request.get("device", "cuda")
        lr = request.get("lr", 0.0001)
        val_split = request.get("val_split", 0.2)

        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, name)

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        load_path = load_path if os.path.exists(load_path) else self.pretrained_model

        # Update/Publish latest model for infer/active learning use
        if os.path.exists(self.final_model) or os.path.islink(self.final_model):
            os.unlink(self.final_model)
        os.symlink(
            os.path.join(os.path.basename(output_dir), "model.pt"),
            self.final_model,
            dir_fd=os.open(self.model_dir, os.O_RDONLY),
        )

        task = MyTrain(
            output_dir=output_dir,
            data_list=self.datastore().datalist(),
            network=self.network,
            load_path=load_path,
            device=device,
            lr=lr,
            val_split=val_split,
        )

        return task(max_epochs=epochs, amp=amp)

    def postproc_label(self, request):
        method = request.get('method')
        task = self.postprocs.get(method)

        # prepare data
        data = {}
        image_file = self._datastore.get_image_uri(request["image"])
        scribbles_file = request.get('scribbles')
        image_name = os.path.basename(image_file).rsplit('.')[0]
        
        data['image'] = image_file
        data['scribbles'] = scribbles_file
        data['logits'] = self.logits_files[image_name]['file']

        # save scribbles/logits if in research mode
        if RESEARCH_MODE:
            self.backup_interaction_data(image_name, data['scribbles'], data['logits'])

        logger.info('\n\timage_name: {}\n\tscribbles: {}\n\tlogits: {}'.format(data['image'], data['scribbles'], data['logits']))

        # run post processing task
        result_file_name, result_json = task(data)
        return {'label': result_file_name, 'params': result_json}

    def backup_interaction_data(self, image_name, scribbles_file, logits_file):
        # To enable future research, we need to save every scribbles instance
        # That is saving all scribbles provided by user, along with the model logits
        # and the name of input image from the given dataset
        
        # load current research database
        with open(self.save_research_data, 'r') as fp:
            backup_data = json.loads(fp.read())

        scrib_ext = ''.join(pathlib.Path(scribbles_file).suffixes)
        logits_ext = ''.join(pathlib.Path(logits_file).suffixes)

        # parse data and see what needs to be done
        if image_name in backup_data.keys():
            current_count = backup_data[image_name]['count']
            new_count = current_count + 1
                        
            scribbles_saved = os.path.join(self.save_research_path, '{}_sc_{}_{}{}'.format(image_name, 
                self.save_research_sessionid, new_count, scrib_ext))
            
            backup_data[image_name]['count'] = new_count
            backup_data[image_name]['scrib_files'] = \
                backup_data[image_name]['scrib_files'] + [os.path.basename(scribbles_saved)]
        else:
            new_count = 1
            scribbles_saved = os.path.join(self.save_research_path, '{}_sc_{}_{}{}'.format(image_name, 
                self.save_research_sessionid, new_count, scrib_ext))
            
            logits_saved = os.path.join(self.save_research_path, '{}_lg_{}{}'.format(image_name, 
                self.save_research_sessionid, logits_ext))
            
            backup_data[image_name] = {}
            backup_data[image_name]['count'] = new_count
            backup_data[image_name]['scrib_files'] = [os.path.basename(scribbles_saved)]
            backup_data[image_name]['logits_file'] = os.path.basename(logits_saved)

        if new_count == 1:
            # copy logits only once
            copyfile(logits_file, logits_saved)

        # copy scribbles everytime       
        copyfile(scribbles_file, scribbles_saved)

        # write data back to data.json
        with open(self.save_research_data, 'w') as fp:
            json.dump(backup_data, fp, indent=4)

    def cleanup_logits_files(self):
        # clean residual logits help from: https://stackoverflow.com/a/32732654
        for key in list(self.logits_files.keys()):
            if key in self.logits_files.keys():
                logger.info(f'removing temp logits file for: {key}')
                cur_item = self.logits_files.pop(key)
                # del file on disk
                os.unlink(cur_item['file'])
