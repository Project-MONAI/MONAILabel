import logging
import os

from lib import (
    MyStrategy,
    MyTrain,
    SegmentationWithWriteLogits,
    SpleenBIFSegCRF,
    SpleenBIFSegGraphCut,
    SpleenBIFSegSimpleCRF,
    SpleenInteractiveGraphCut,
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

        super().__init__(app_dir, studies, os.path.join(self.model_dir, "train_stats.json"))

    def init_infers(self):
        infers = {
            "Spleen_Segmentation": SegmentationWithWriteLogits(
                self.final_model, load_from_mmar(self.mmar, self.model_dir)
            ),
            "BIFSeg+CRF": SpleenBIFSegCRF(),
            "BIFSeg+SimpleCRF": SpleenBIFSegSimpleCRF(),
            "BIFSeg+GraphCut": SpleenBIFSegGraphCut(),
            "Int.+BIFSeg+GraphCut": SpleenInteractiveGraphCut(),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }

    def infer(self, request, datastore=None):
        image = request.get("image")

        # add saved logits into request
        if self._infers[request.get("model")].type == InferType.SCRIBBLE:
            saved_labels = self.datastore().get_labels_by_image_id(image)
            for label, tag in saved_labels.items():
                if tag == "logits":
                    request["logits"] = self.datastore().get_label_uri(label)
            logger.info(f"Updated request: {request}")

        result = super().infer(request)
        result_params = result.get("params")

        # save logits
        logits = result_params.get("logits")
        if logits:
            self.datastore().save_label(image, logits, "logits")
            os.unlink(logits)

        result_params.pop("logits", None)
        logger.info(f"Final Result: {result}")
        return result

    def train(self, request):
        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, request.get("name", "model_01"))

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        if not os.path.exists(load_path) and request.get("pretrained", True):
            load_path = None
            network = load_from_mmar(self.mmar, self.model_dir)
        else:
            network = load_from_mmar(self.mmar, self.model_dir, pretrained=False)

        # Datalist for train/validation
        train_d, val_d = self.partition_datalist(self.datastore().datalist(), request.get("val_split", 0.2))

        task = MyTrain(
            output_dir=output_dir,
            train_datalist=train_d,
            val_datalist=val_d,
            network=network,
            load_path=load_path,
            publish_path=self.final_model,
            stats_path=self.train_stats_path,
            device=request.get("device", "cuda"),
            lr=request.get("lr", 0.0001),
            val_split=request.get("val_split", 0.2),
            max_epochs=request.get("epochs", 1),
            amp=request.get("amp", True),
            train_batch_size=request.get("train_batch_size", 1),
            val_batch_size=request.get("val_batch_size", 1),
        )
        return task()
