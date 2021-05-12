import io
import json
import os
import pathlib
import shutil
from datetime import datetime
from typing import Any, Dict, List, Union

from pydantic import BaseModel

from monailabel.interfaces.datastore import Datastore, LabelTag
from monailabel.interfaces.exception import ImageNotFoundException, LabelNotFoundException


class ImageModel(BaseModel):
    id: str
    info: Dict[str, Any] = {}


class LabelModel(BaseModel):
    id: str
    tag: str
    info: Dict[str, Any] = {}


class ObjectModel(BaseModel):
    image: ImageModel
    labels: List[LabelModel] = []


class LocalDatastoreModel(BaseModel):
    name: str
    description: str
    objects: List[ObjectModel] = []


class LocalDatastore(Datastore):
    """
    Class to represent a datastore local to the MONAI-Label Server

    Attributes
    ----------
    `name: str`
        The name of the datastore

    `description: str`
        The description of the datastore
    """

    def __init__(self, datastore_path: str, datastore_config: str = "datastore.json"):
        """
        Creates a `LocalDataset` object

        Parameters:

        `datastore_path: str`
            a string to the directory tree of the desired dataset

        `datastore_config: str`
            optional file name of the dataset configuration file (by default `dataset.json`)
        """
        self._datastore_path = datastore_path
        self._datastore_config_path = os.path.join(datastore_path, datastore_config)

        # check if dataset configuration file exists
        if os.path.exists(self._datastore_config_path):
            self._datastore = LocalDatastoreModel.parse_file(self._datastore_config_path)
        else:
            self._datastore = LocalDatastoreModel(name="new-dataset", description="New Dataset")

            files = LocalDatastore._list_files(datastore_path)

            for file in files:
                self._datastore.objects.append(ObjectModel(image=ImageModel(id=file)))

            self._update_datastore_file()

    @property
    def name(self) -> str:
        """
        Dataset name (if one is assigned)

        Returns:
            name (str): Dataset name as string
        """
        return self._datastore.name

    @name.setter
    def name(self, name: str):
        """
        Sets the dataset name in a standardized format (lowercase, no spaces).

            Parameters:
                name (str): Desired dataset name
        """
        self._datastore.name = name
        self._update_datastore_file()

    @property
    def description(self) -> str:
        """
        Gets the description field for the dataset

        :return description: str
        """
        return self._datastore.description

    @description.setter
    def description(self, description: str):
        """
        Set a description for the dataset

        :param description: str
        """
        self._datastore.description = description
        self._update_datastore_file()

    def datalist(self, full_path=True) -> List[Dict[str, str]]:

        items = []
        for obj in self._datastore.objects:
            image_path = self._get_path(obj.image.id, full_path)
            for label in obj.labels:
                if label.tag == LabelTag.FINAL.value:
                    items.append(
                        {
                            "image": image_path,
                            "label": self._get_path(label.id, full_path),
                        }
                    )
        return items

    def get_image(self, image_id: str) -> io.BytesIO:

        buf = None
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                with open(os.path.join(self._datastore_path, obj.image.id), "rb") as f:
                    buf = io.BytesIO(f.read())
                break
        return buf

    def get_image_uri(self, image_id: str) -> str:

        image_path = None
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                image_path = os.path.join(self._datastore_path, obj.image.id)
                break
        return image_path

    def get_image_info(self, image_id: str) -> Dict[str, Any]:

        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                return obj.image.info

        return {}

    def get_label(self, label_id: str) -> io.BytesIO:

        buf = None
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    with open(os.path.join(self._datastore_path, label.id)) as f:
                        buf = io.BytesIO(f.read())
        return buf

    def get_label_uri(self, label_id: str) -> str:

        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    return os.path.join(self._datastore_path, label.id)
        return None

    def get_labels_by_image_id(self, image_id: str) -> LabelModel:

        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                labels = {label.id: label.tag for label in obj.labels}
                return labels
        return {}

    def get_label_info(self, label_id: str) -> Dict[str, Any]:

        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    return label.info

        return {}

    def get_labeled_images(self) -> List[str]:

        image_ids = []
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.tag == LabelTag.FINAL.value:
                    image_ids.append(obj.image.id)

        return image_ids

    def get_unlabeled_images(self) -> List[str]:

        image_ids = []
        for obj in self._datastore.objects:
            if not obj.labels:
                image_ids.append(obj.image.id)

            for label in obj.labels:
                if label.tag != LabelTag.FINAL.value:
                    image_ids.append(obj.image.id)

        return image_ids

    def list_images(self) -> List[str]:

        return [obj.image.id for obj in self._datastore.objects]

    def save_label(self, image_id: str, label_filename: str, label_tag: LabelTag) -> str:

        for obj in self._datastore.objects:

            if obj.image.id == image_id:

                image_ext = "".join(pathlib.Path(image_id).suffixes)
                label_ext = "".join(pathlib.Path(label_filename).suffixes)
                label_id = "label_" + label_tag.value + "_" + image_id.replace(image_ext, "") + label_ext

                datastore_label_path = os.path.join(self._datastore_path, label_id)
                shutil.copy(src=label_filename, dst=datastore_label_path, follow_symlinks=True)

                if label_tag.value not in [label.tag for label in obj.labels]:
                    obj.labels.append(
                        LabelModel(
                            id=label_id,
                            tag=label_tag.value,
                        )
                    )
                else:
                    for label_index, label in enumerate(obj.labels):
                        if label.tag == label_tag.value:
                            obj.labels[label_index] = LabelModel(id=label_id, tag=label_tag.value)

                self._update_datastore_file()

                return label_id

        else:
            raise ImageNotFoundException(f"Image {image_id} not found")

    def update_image_info(self, image_id: str, info: Dict[str, Any]) -> None:

        for obj_index, obj in enumerate(self._datastore.objects):
            if obj.image.id == image_id:
                self._datastore.objects[obj_index].image.info.update(info)
                break
        else:
            raise ImageNotFoundException(f"Image {image_id} not found")

    def update_label_info(self, label_id: str, info: Dict[str, Any]) -> None:

        for obj_index, obj in enumerate(self._datastore.objects):
            for label_index, label in enumerate(obj.labels):
                if label.id == label_id:
                    self._datastore.objects[obj_index].labels[label_index].info.update(info)
                    return

        raise LabelNotFoundException(f"Label {label_id} not found")

    @staticmethod
    def _list_files(path: str):
        relative_file_paths = []
        for root, dirs, files in os.walk(path):
            base_dir = root.strip(path)
            relative_file_paths.extend([os.path.join(base_dir, file) for file in files])
        return relative_file_paths

    def _update_datastore_file(self):
        with open(self._datastore_config_path, "w") as f:
            json.dump(self._datastore.dict(), f, indent=2, default=str)

    def _get_path(self, path: str, full_path=True):
        if not full_path or os.path.isabs(path):
            return path
        return os.path.realpath(os.path.join(self._datastore_path, path))
