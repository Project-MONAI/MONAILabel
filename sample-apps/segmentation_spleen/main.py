import json
import logging
import os
from json import JSONDecodeError

import yaml
from lib import MyInfer, MyTrain
from lib.activelearning import MyStrategy, Tta
from monai.apps import load_from_mmar

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.scoring.tta_scoring import TtaScoring

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.mmar = "clara_pt_spleen_ct_segmentation_1"

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Segmentation - Spleen",
            description="Active Learning solution to label Spleen Organ over 3D CT Images",
            version=2,
        )

    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_trainers(self):
        return {
            "segmentation_spleen": MyTrain(
                self.model_dir, load_from_mmar(self.mmar, self.model_dir), publish_path=self.final_model
            )
        }

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
            "Tta": Tta(),
        }

    def init_scoring_methods(self):
        return {
            "tta_scoring": TtaScoring(),
        }

    # TODO:: This will be removed once DICOM Web support is added through datastore
    def infer(self, request, datastore=None):
        image = request["image"]
        try:
            dicom = json.loads(image)
            logger.info(f"Temporary Hack:: Looking mapped image for: {dicom}")
            with open(os.path.join(os.path.dirname(__file__), "dicom.yaml"), "r") as fc:
                meta = yaml.full_load(fc)["series"]
                request["image"] = meta[dicom["SeriesInstanceUID"]]
                logger.info(f"Using Image: {request['image']}")
        except JSONDecodeError:
            pass
        return super().infer(request, datastore)

    # TODO:: This will be removed once DICOM Web support is added through datastore
    def next_sample(self, request):
        res = super().next_sample(request)
        try:
            with open(os.path.join(os.path.dirname(__file__), "dicom.yaml"), "r") as fc:
                meta = {v: k for k, v in yaml.full_load(fc)["studies"].items()}
                res["id"] = meta[res["id"]]
                logger.info(f"Using studies: {res['id']}")
        except JSONDecodeError:
            pass
        return res
