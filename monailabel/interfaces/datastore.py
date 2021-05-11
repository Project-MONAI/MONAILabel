from abc import ABCMeta, abstractmethod
from typing import Dict, List


class LabelTag:
    TEMP: str = "label_temp"
    FINAL: str = "label"


class Datastore(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get Name of Datastore
        """
        pass

    @name.setter
    @abstractmethod
    def name(self, name: str):
        """
        Set Name for Datastore

        :param name: Name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get Description of Datastore
        """
        pass

    @description.setter
    @abstractmethod
    def description(self, description: str):
        """
        Set Description for Datastore

        :param description: Description
        """
        pass

    @abstractmethod
    def to_json(self) -> Dict:
        """
        Json representation of the datastore
        """
        pass

    @abstractmethod
    def get_images(self) -> List[str]:
        """
        List all the image ids in the dataset
        """
        pass

    @abstractmethod
    def get_image(self, id: str) -> Dict:
        """
        Get Image Info/Details for matching image id
        """
        pass

    @abstractmethod
    def get_labels(self, tag: str = LabelTag.FINAL) -> List[str]:
        """
        List all the labels ids in the dataset for the matching tag

        :param tag: Tag for matching
        """
        pass

    @abstractmethod
    def get_label(self, id: str) -> Dict:
        """
        Get Image Info/Details for matching label id
        """
        pass

    @abstractmethod
    def get_label_by_image_id(self, image_id: str, tag: str = LabelTag.FINAL) -> str:
        """
        Get Label Id for matching image id and tag

        :param image_id: Image ID
        :param tag: Label Tag
        """
        pass

    @abstractmethod
    def get_unlabeled_images(self, tag: str = LabelTag.FINAL) -> List[str]:
        """
        Get All Unlabeled images (non existing labels for given tag)

        :param tag: Label Tag
        """
        pass

    def save_label(self, image_id: str, label_file: str, tag: str = LabelTag.FINAL, info: Dict = {}) -> str:
        """
        Save Label

        :param image_id: Image Id
        :param label_file: Label File
        :param tag: Label Tag
        :param info: Label Info
        """
        pass

    @abstractmethod
    def update_label_info(self, id: str, info: Dict, tag: str = LabelTag.FINAL, override: bool = False) -> None:
        """
        Update/Override Label Info

        :param id: Label Id
        :param info: Label Info
        :param tag: Label Tag
        :param override: Override existing Info with new
        """
        pass

    @abstractmethod
    def datalist(self, full_path=True, tag: str = LabelTag.FINAL) -> List[Dict]:
        """
        Get Datalist JSON for training

        :param full_path: Add Full path for images/labels
        :param tag: Label Tag
        :return: Dictionary of image/label pairs
        """
        pass

    @abstractmethod
    def get_path(self, id) -> str:
        """
        Get Download path the id

        :param id: image or label id
        :return: path
        """
        pass
