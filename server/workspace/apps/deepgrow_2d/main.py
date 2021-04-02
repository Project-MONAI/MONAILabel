import logging
import os

from lib import DeepgrowInferenceEngine
from server.interface import ServerException, ServerError
from server.interface.app import MONAIApp

logger = logging.getLogger(__name__)


class DeepgrowApp(MONAIApp):
    def __init__(self, name, app_dir):
        super().__init__(name=name, app_dir=app_dir)
        self.inference_engine = None

    def infer(self, request):
        if self.inference_engine is None:
            model = os.path.join(self.app_dir, 'model', 'model.ts')
            if os.path.exists(model):
                self.inference_engine = DeepgrowInferenceEngine(model)

        if self.inference_engine is None:
            raise ServerException(
                ServerError.INFERENCE_ERROR,
                "Inference Engine is not Initialized. There is no pre-trained model available"
            )

        image = request['image']
        params = request.get('params')
        device = request.get('device', 'cuda')

        result_file_name, result_json = self.inference_engine.run(image, params, device)
        return result_file_name, result_json

    def train(self, request):
        raise Exception("Not Implemented")
