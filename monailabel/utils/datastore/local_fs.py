import copy
import fnmatch
import json
import logging
import os
import pathlib
import shutil
import time
from typing import Dict, List

from pydantic import BaseModel

from monailabel.interfaces.datastore import Datastore, LabelTag
from monailabel.interfaces.exception import ImageNotFoundException

logger = logging.getLogger(__name__)


class BaseObject(BaseModel):
    id: str
    info: Dict = {}


class ObjectModel(BaseModel):
    image: BaseObject
    labels: Dict[str, BaseObject] = {}


class LocalDatastoreModel(BaseModel):
    name: str
    description: str
    objects: List[ObjectModel] = []


class LocalDatastore(Datastore):
    """
    Class to represent a datastore local to the MONAILabel Server

    Attributes
    ----------
    `name: str`
        The name of the datastore

    `description: str`
        The description of the datastore
    """

    def __init__(self, datastore_path: str, datastore_config: str = "datastore.json", pattern="*.nii.gz"):
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
            logger.info(f"Using Datastore Path: {datastore_path}")

            files = os.listdir(datastore_path)
            files = fnmatch.filter(files, pattern) if pattern else files
            logger.info("Following Images will be added into dataset: [{}]".format(", ".join(files)))

            for file in files:
                self._datastore.objects.append(ObjectModel(image=BaseObject(id=file)))

            self._update_datastore_file()

    @property
    def name(self) -> str:
        return self._datastore.name

    @name.setter
    def name(self, name: str):
        self._datastore.name = name
        self._update_datastore_file()

    @property
    def description(self) -> str:
        return self._datastore.description

    @description.setter
    def description(self, description: str):
        self._datastore.description = description
        self._update_datastore_file()

    def to_json(self) -> Dict:
        return self._datastore.dict()

    def get_images(self) -> List[str]:
        return [obj.image.id for obj in self._datastore.objects]

    def get_image(self, id: str) -> Dict:
        for obj in self._datastore.objects:
            if obj.image.id == id:
                return self._get_info(id, obj.image.info)
        return {}

    def get_labels(self, tag: str = LabelTag.FINAL) -> List[str]:
        for obj in self._datastore.objects:
            if obj.labels.get(tag):
                return [label.id for label in obj.labels.get(tag)]
        return []

    def get_label(self, id: str, tag: str = LabelTag.FINAL) -> Dict:
        for obj in self._datastore.objects:
            label = obj.labels.get(tag)
            if label and label.id == id:
                return self._get_info(id, label.info)
        return {}

    def get_label_by_image_id(self, image_id: str, tag: str = LabelTag.FINAL) -> str:
        for obj in self._datastore.objects:
            if obj.image.id == image_id and obj.labels.get(tag):
                return obj.labels.get(tag).id
        return None

    def get_unlabeled_images(self, tag: str = LabelTag.FINAL) -> List[str]:
        return [obj.image.id for obj in self._datastore.objects if not obj.labels.get(tag)]

    def save_label(self, image_id: str, label_file: str, tag: str = LabelTag.FINAL, info: Dict = {}) -> str:
        for obj in self._datastore.objects:
            if image_id == obj.image.id:
                base_name = image_id.rsplit(".")[0]
                file_ext = "".join(pathlib.Path(label_file).suffixes)
                label_id = f"{tag}_{base_name}{file_ext}"

                logger.info(f"Saving Label: {label_id} with tag: {tag} against image: {image_id}")
                shutil.copy(label_file, os.path.join(self._datastore_path, label_id))
                # If tag exists, update info

                info.update({"timestamp": int(time.time())})
                if obj.labels.get(tag):
                    obj.labels.get(tag).info.update(info)
                else:
                    obj.labels[tag] = BaseObject(id=label_id, info=info)

                self._update_datastore_file()
                return label_id

        raise ImageNotFoundException(f"Image {image_id} not found")

    def update_label_info(self, id: str, info: Dict, tag: str = LabelTag.FINAL, override: bool = False) -> None:
        info = self.get_label(id)
        if override:
            info.clear()
        if info:
            info.update(info)
        self._update_datastore_file()

    def datalist(self, full_path=True, tag: str = LabelTag.FINAL) -> List[Dict]:
        items = []
        for obj in self._datastore.objects:
            labels = {"label": obj.labels.get(tag)} if tag else obj.labels
            for k, v in labels.items():
                if not v:
                    continue

                item = {"image": self._get_path(obj.image.id, full_path)}
                if obj.image.info:
                    item["image_info"] = obj.image.info
                item[k] = self._get_path(v.id, full_path)
                if v.info:
                    item[f"{k}_info"] = v.info
                items.append(item)
        return items

    def get_path(self, id) -> str:
        return self._get_path(id, full_path=True)

    def _update_datastore_file(self):
        with open(self._datastore_config_path, "w") as f:
            json.dump(self._datastore.dict(), f, indent=2, default=str)

    def _get_path(self, path: str, full_path=True):
        if not full_path or os.path.isabs(path):
            return path
        return os.path.realpath(os.path.join(self._datastore_path, path))

    def _get_info(self, id, info):
        d = copy.deepcopy(info)
        d["path"] = self.get_path(id)
        return d
