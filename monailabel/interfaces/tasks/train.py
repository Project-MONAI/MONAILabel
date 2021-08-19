from abc import ABCMeta, abstractmethod

from monailabel.interfaces.datastore import Datastore


class TrainTask(metaclass=ABCMeta):
    """
    Basic Train Task
    """

    def __init__(self, description):
        self.description = description

    def info(self):
        return {"description": self.description, "config": self.config()}

    def config(self):
        return {}

    def stats(self):
        return {}

    @abstractmethod
    def __call__(self, request, datastore: Datastore):
        pass
