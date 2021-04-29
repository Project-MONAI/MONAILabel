import logging

from monailabel.interface import ActiveLearning
from monailabel.interface import Datastore

logger = logging.getLogger(__name__)


class MyActiveLearning(ActiveLearning):
    def __call__(self, request, datastore: Datastore):
        return super().__call__(request, datastore)
