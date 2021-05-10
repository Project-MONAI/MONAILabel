import logging

from lib import MyActiveLearning, MyInfer, MyTrain

from monailabel.interfaces.app import MONAILabelApp

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        super().__init__(app_dir=app_dir, studies=studies)

    def infer(self, request):
        return MyInfer()(request)

    def train(self, request):
        return MyTrain()(request)

    def next_sample(self, request):
        return MyActiveLearning()(request)

    def save_label(self, request):
        return super().save_label(request)
