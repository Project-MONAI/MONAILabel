from abc import ABCMeta, abstractmethod
from datetime import datetime
import io
from pydantic import BaseModel
from typing import List, Union, Dict, Optional


class ImageModel(BaseModel):
    id: str
    access_timestamp: Optional[datetime]


class ScoreModel(BaseModel):
    name: str
    value: Union[int, float]


class OriginalLabelModel(BaseModel):
    id: str
    access_timestamp: datetime


class UpdatedLabelModel(OriginalLabelModel):
    scores: Optional[List[ScoreModel]]


class LabelSetModel(BaseModel):
    original: Optional[OriginalLabelModel]
    updated: Optional[UpdatedLabelModel]


class ObjectModel(BaseModel):
    image: Optional[ImageModel]
    labels: Optional[LabelSetModel]


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
    def datalist(self, full_path=True, flatten_labels=True) -> List[Dict[str, Dict[str, str]]]: pass

    @abstractmethod
    def get_labels_by_image_id(self, image_id: str) -> List[LabelSetModel]: pass

    @abstractmethod
    def get_label_scores(self, image_id: str) -> List[ScoreModel]: pass

    @abstractmethod
    def get_unlabeled_images(self) -> List[str]: pass

    @abstractmethod
    def list_images(self) -> List[ImageModel]: pass

    @abstractmethod
    def list_labels(self) -> List[LabelSetModel]: pass

    @abstractmethod
    def set_original_label(self, image: str, label: io.BytesIO) -> None: pass

    @abstractmethod
    def save_updated_label(self, image: str, label: io.BytesIO, scores: List[ScoreModel]=[]) -> str: pass

    @abstractmethod
    def set_label_score(self, label_id: str, score_name: str, score_value: Union[int, float]) -> None: pass

