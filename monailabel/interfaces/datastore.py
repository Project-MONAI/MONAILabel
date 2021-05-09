from enum import Enum
import io
from abc import ABCMeta, abstractmethod
from typing import List, Union, Dict


class LabelStage(Enum):
    ORIGINAL='original'
    SAVED='saved'


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

    @abstractmethod
    def datalist(self, full_path=True, flatten_labels=True) -> list: pass

    @abstractmethod
    def find_data_by_image(self, pattern: str) -> list: pass

    @abstractmethod
    def find_data_by_label(self, pattern: str) -> list: pass

    @abstractmethod
    def get_label_scores(self, label_id: str) -> List[Dict[str, Union[int, float]]]: pass

    @abstractmethod
    def get_unlabeled_images(self) -> list: pass

    @abstractmethod
    def list_images(self) -> list: pass

    @abstractmethod
    def list_labels(self) -> list: pass

    @abstractmethod
    def save_label(self, image: str, label_stage: LabelStage, label_id: str, label: io.BytesIO, scores: List[Dict[str, Union[int, float]]]=[]) -> str: pass

    @abstractmethod
    def set_label_score(self, label_id: str, score_name: str, score_value: Union[int, float]) -> None: pass

