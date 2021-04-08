import logging
import os

import yaml

from lib import SpleenTrainEngine, SegmentationSpleen, Deepgrow2D, Deepgrow3D
from server.interface import ServerException, ServerError, InferenceEngine
from server.interface.app import MONAIApp

logger = logging.getLogger(__name__)


class SpleenApp(MONAIApp):
    def __init__(self, app_dir, cache=True):
        super().__init__(app_dir=app_dir)

        self.studies = os.path.join(app_dir, "studies")
        self.cached = {} if cache else None
        self.infer_models = {
            "deepgrow_2d": (Deepgrow2D, "deepgrow_2d.ts"),
            "deepgrow_3d": (Deepgrow3D, "deepgrow_2d.ts"),
            "segmentation_spleen": (SegmentationSpleen, "segmentation_spleen.ts"),
            "model": (SegmentationSpleen, "model.ts")
        }

    def info(self):
        with open(os.path.join(self.app_dir, "info.yaml"), 'r') as fc:
            return yaml.full_load(fc)

    # TODO:: Define the definition/schema for infer request
    def infer(self, request):
        model_name = request.get('model')
        model_name = model_name if model_name else 'model'

        engine = self.cached.get(model_name) if self.cached is not None else None
        if engine is None:
            if self.infer_models.get(model_name) is None:
                raise ServerException(
                    ServerError.INFERENCE_ERROR,
                    f"Inference Engine for Model '{model_name}' Not Found"
                )

            c, model_file = self.infer_models[model_name]
            model_file = os.path.join(self.app_dir, 'model', model_file)
            if os.path.exists(model_file):
                engine: InferenceEngine = c(model_file)

        if engine is None:
            raise ServerException(
                ServerError.INFERENCE_ERROR,
                "Inference Engine is not Initialized. There is no pre-trained model available"
            )

        if self.cached is not None:
            self.cached[model_name] = engine

        image = request['image']
        image = image if os.path.isabs(image) else os.path.join(self.studies, image)
        params = request.get('params')
        device = request.get('device', 'cuda')

        result_file_name, result_json = engine.run(image, params, device)
        return {"label": result_file_name, "params": result_json}

    # TODO:: Define the definition/schema for train request
    def train(self, request):
        params = {
            'output_dir': os.path.join(self.app_dir, "model", "train_0"),  # TODO:: Find next train_x
            'data_list': os.path.join(self.studies, "dataset.json"),
            'data_root': self.studies
        }
        for p in params:
            if request.get(p) is None:
                request[p] = params[p]

        epochs = request['epochs']
        amp = request.get('amp', False)

        logger.info(f"Training request: {request}")
        engine = SpleenTrainEngine(request)
        stats = engine.run(max_epochs=epochs, amp=amp)
        return stats

    def stop_train(self, request):
        return {"status": "NOT IMPLEMENTED YET"}

    def next_sample(self, request):
        return {"image": os.path.join(self.studies, "imagesTr/spleen_2.nii.gz")}

    def save_label(self, request):
        return {
            "image": os.path.join(self.studies, "imagesTr/spleen_2.nii.gz"),
            "label": os.path.join(self.studies, "labelsTr/spleen_2.nii.gz")
        }
