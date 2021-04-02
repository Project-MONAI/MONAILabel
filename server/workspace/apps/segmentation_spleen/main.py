import logging
import os

from lib import SpleenTrainEngine, SpleenInferenceEngine
from server.interface import ServerException, ServerError
from server.interface.app import MONAIApp

logger = logging.getLogger(__name__)


# TODO:: One Advantage of inheriting MONAIApp.. is bringing more users to use MONAI/MONAI-specific... business decision?
# TODO:: MONAIApp - can be a new term? And let users develop more of them.

class SpleenApp(MONAIApp):
    def __init__(self, name, app_dir):
        super().__init__(name=name, app_dir=app_dir)
        self.inference_engine = None

    # TODO:: Define the definition for infer request
    # Example:: request = {'image': 'image_path', 'params': {}}
    def infer(self, request):
        if self.inference_engine is None:
            model = os.path.join(self.app_dir, 'model', 'model.ts')
            if os.path.exists(model):
                self.inference_engine = SpleenInferenceEngine(model)

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

    # TODO:: Define the definition/schema for train request
    def train(self, request):
        epochs = request['epochs']
        amp = request.get('amp', False)

        logger.info(f"Training request: {request}")
        engine = SpleenTrainEngine(request)
        stats = engine.run(max_epochs=epochs, amp=amp)
        return stats
