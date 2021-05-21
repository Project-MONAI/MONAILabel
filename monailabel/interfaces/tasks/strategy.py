from abc import ABCMeta, abstractmethod

from monailabel.interfaces.datastore import Datastore


class Strategy(metaclass=ABCMeta):
    """
    Basic Active Learning Strategy
    """

    def __init__(self, description):
        self.description = description

    def info(self):
        return {
            "description": self.description,
        }

    @abstractmethod
    def __call__(self, request, datastore: Datastore):
        pass
