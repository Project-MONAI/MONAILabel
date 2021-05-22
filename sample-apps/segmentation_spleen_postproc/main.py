import atexit
import logging
import os

from lib import MyInfer, MyStrategy, MyTrain, SpleenCRF
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monailabel.interfaces import MONAILabelApp
from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


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
            "CRF": SpleenCRF(method="CRF"),
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

        if "logits_file_name" in result_json:
            logits_file = result_json.pop("logits_file_name")
            logits_json = result_json.pop("logits_json")
            self.logits_files[os.path.basename(request.get("image")).rsplit(".")[0]] = {
                "file": logits_file,
                "params": logits_json,
            }
            logger.info(f"Logits files saved: {self.logits_files}")

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
        method = request.get("method")
        task = self.postprocs.get(method)

        # prepare data
        data = {}
        image_file = self._datastore.get_image_uri(request["image"])
        scribbles_file = request.get("scribbles")
        image_name = os.path.basename(image_file).rsplit(".")[0]

        data["image"] = image_file
        data["scribbles"] = scribbles_file
        data["logits"] = self.logits_files[image_name]["file"]

        logger.info(
            "\n\timage_name: {}\n\tscribbles: {}\n\tlogits: {}".format(data["image"], data["scribbles"], data["logits"])
        )

        # run post processing task
        result_file_name, result_json = task(data)
        return {"label": result_file_name, "params": result_json}

    def cleanup_logits_files(self):
        for key in list(self.logits_files.keys()):
            if key in self.logits_files.keys():
                logger.info(f"removing temp logits file for: {key}")
                cur_item = self.logits_files.pop(key)
                # del file on disk
                os.unlink(cur_item["file"])
