import logging
import os

from lib import (
    MyStrategy,
    MyTrain,
    SegmentationWithWriteLogits,
    SpleenInteractiveGraphCut,
    SpleenISegCRF,
    SpleenISegGraphCut,
    SpleenISegSimpleCRF,
)
from monai.apps import load_from_mmar

from monailabel.interfaces import MONAILabelApp
from monailabel.interfaces.tasks import InferType
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.mmar = "clara_pt_spleen_ct_segmentation_1"

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Segmentation - Spleen + Scribbles",
            description="Active Learning solution to label Spleen Organ over 3D CT Images.  "
            "It includes multiple scribbles method that incorporate user scribbles to improve labels",
            version=2,
        )

    def init_infers(self):
        infers = {
            "Spleen_Segmentation": SegmentationWithWriteLogits(
                self.final_model, load_from_mmar(self.mmar, self.model_dir)
            ),
            "ISeg+GraphCut": SpleenISegGraphCut(),
            "ISeg+CRF": SpleenISegCRF(),
            "ISeg+SimpleCRF": SpleenISegSimpleCRF(),
            "ISeg+InteractiveGraphCut": SpleenInteractiveGraphCut(),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_trainers(self):
        return {"segmentation_spleen": MyTrain(self.model_dir, load_from_mmar(self.mmar, self.model_dir))}

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }

    def infer(self, request, datastore=None):
        image = request.get("image")

        # add saved logits into request
        if self._infers[request.get("model")].type == InferType.SCRIBBLES:
            saved_labels = self.datastore().get_labels_by_image_id(image)
            for label, tag in saved_labels.items():
                if tag == "logits":
                    request["logits"] = self.datastore().get_label_uri(label)
            logger.info(f"Updated request: {request}")

        result = super().infer(request)
        result_params = result.get("params")

        # save logits
        logits = result_params.get("logits")
        if logits and self._infers[request.get("model")].type == InferType.SEGMENTATION:
            self.datastore().save_label(image, logits, "logits")
            os.unlink(logits)

        result_params.pop("logits", None)
        logger.info(f"Final Result: {result}")
        return result
