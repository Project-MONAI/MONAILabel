import io
from abc import ABCMeta, abstractmethod


class Datastore(metaclass=ABCMeta):

    @property
    @abstractmethod
    def name(self) -> str: pass

    @name.setter
    @abstractmethod
    def name(self, name: str): pass

    @property
    @abstractmethod
    def description(self) -> str: pass

    @description.setter
    @abstractmethod
    def description(self, description: str): pass

    @property
    @abstractmethod
    def info(self): pass

    @abstractmethod
    def update_label_info(self, label_id: str, description: str) -> dict: pass

    @abstractmethod
    def list_images(self) -> list: pass

    @abstractmethod
    def list_labels(self) -> list: pass

    @abstractmethod
    def find_objects(self, pattern: str, match_label: bool) -> list: pass

    @abstractmethod
    def save_label(self, image: str, label_id: str, label: io.BytesIO) -> str: pass

    @abstractmethod
    def get_unlabeled_images(self) -> list: pass

    @abstractmethod
    def datalist(self, full_path=True, flatten_labels=True) -> list: pass
