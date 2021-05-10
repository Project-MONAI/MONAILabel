import logging

from monailabel.interfaces import ActiveLearning, Datastore

logger = logging.getLogger(__name__)


class MyActiveLearning(ActiveLearning):
    def __call__(self, request, datastore: Datastore):
        return super().__call__(request, datastore)
