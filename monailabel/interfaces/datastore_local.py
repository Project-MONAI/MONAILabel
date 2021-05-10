from datetime import datetime
import io
import json
import os
import pathlib
from pydantic import BaseModel
from typing import List, Union, Dict, Optional


from monailabel.interfaces.datastore import Datastore, ImageModel, LabelSetModel, ObjectModel, ScoreModel, OriginalLabelModel, UpdatedLabelModel
from monailabel.interfaces.exception import ImageNotFoundException



class LocalDatastoreModel(BaseModel):
    name: str
    description: str
    objects: Optional[List[ObjectModel]]


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
    def __init__(self, datastore_path: str, datastore_config: str = 'dataset.json'):
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
            self._datastore = LocalDatastoreModel(name="new-dataset",
                                                  description="New Dataset")

            files = LocalDatastore._list_files(datastore_path)
            self._datastore.objects = []
            for file in files:
                self._datastore.objects.append(
                    ObjectModel(image=ImageModel(id=file))
                )

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
        """
        return self._datastore.description

    @description.setter
    def description(self, description: str):
        """
        Set a description for the dataset
        """
        self._datastore.description = description
        self._update_datastore_file()

    def list_images(self) -> List[ImageModel]:
        """
        List all the images in the dataset
        """
        return [obj.image for obj in self._datastore.objects]

    def list_labels(self) -> List[LabelSetModel]:
        """
        List all the labels in the dataset
        """
        return [obj.labels for obj in self._datastore.objects]

    def get_labels_by_image_id(self, image_id: str) -> LabelSetModel:
        """
        Return the labels for the given image id

        Parameters:

        `image_id: str`
            a string that matches the image_id in the datastore

        Returns:

        the labels associated to the image
        """
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                return obj.labels
        return None

    def get_unlabeled_images(self) -> List[str]:
        """
        Get a list of all the unlabeled_images without an associated label

        Returns:

            a list of image paths that do not have corresponding labels
        """
        unlabeled_images = []
        for obj in self._datastore.objects:
            if obj.labels is None:
                unlabeled_images.append(os.path.join(self._datastore_path, obj.image.id))
        return unlabeled_images

    def _save_label_helper(self, image_id: str, is_original_label: bool, label: io.BytesIO, scores: List[ScoreModel]) -> None:

        for object_index, obj in enumerate(self._datastore.objects):

            image_id = image_id.replace(self._datastore_path, '').lstrip('/')
            if image_id == obj.image.id:

                if not self._datastore.objects[object_index].labels:
                    self._datastore.objects[object_index].labels = LabelSetModel()

                file_ext = ''.join(pathlib.Path(image_id).suffixes)

                image_base_path = image_id.replace(file_ext, '')
                labels_path = os.path.join(self._datastore_path, image_base_path)
                os.makedirs(labels_path, exist_ok=True)

                if is_original_label:
                    label_id = os.path.join(image_base_path, 'original', file_ext)

                    self._datastore.objects[object_index].labels.original = OriginalLabelModel(id = label_id,
                                                                                               access_timestamp = datetime.utcnow())
                else:
                    label_id = os.path.join(image_base_path, 'updated'+file_ext)

                    self._datastore.objects[object_index].labels.updated = UpdatedLabelModel(id = label_id,
                                                                                             scores = scores,
                                                                                             access_timestamp = datetime.utcnow())

                with open(os.path.join(self._datastore_path, label_id), 'wb') as f:
                    label.seek(0)
                    f.write(label.getbuffer())

                self._update_datastore_file()

                return label_id

        raise ImageNotFoundException(f"Image {image_id} not found")

    def set_original_label(self, image_id: str, label: io.BytesIO) -> None:
        """
        Save the original/autogenerated label for the given image in the dataset

        Parameters:

        `image: str`
            the path of the image file in the dataset to which the provided `label_id` and `label` should be associated

        `label: io.BytesIO`
            a byte stream of the file to be saved in the dataset
        """
        self._save_label_helper(image_id, True, label, None)

    def save_updated_label(self, image_id: str, label: io.BytesIO, scores: List[ScoreModel]=None) -> str:
        """
        Save the updated label for the given image in the dataset

        Parameters:

        `image: str`
            the path of the image file in the dataset to which the provided `label_id` and `label` should be associated

        `label: io.BytesIO`
            a byte stream of the file to be saved in the dataset

        `scores: List[ScoreModel]=None`
            a list of scores to attibute to the saved/updated/corrected label
        """
        self._save_label_helper(image_id, False, label, scores)

    @staticmethod
    def _list_files(path: str):
        relative_file_paths = []
        for root, dirs, files in os.walk(path):
            base_dir = root.strip(path)
            relative_file_paths.extend([os.path.join(base_dir, file) for file in files])
        return relative_file_paths

    def _update_datastore_file(self):
        with open(self._datastore_config_path, 'w') as f:
            json.dump(self._datastore.dict(), f, indent=2, default=str)

    def _get_path(self, path: str, full_path=True):
        if not full_path or os.path.isabs(path):
            return path
        return os.path.realpath(os.path.join(self._datastore_path, path))

    def datalist(self, full_path=True) -> List[Dict[str, Dict[str, str]]]:
        """
        Get Data List of image and valid/existing label pairs

        :param full_path: Add full path for image/label objects
        :return: List of dictionary objects.  Each dictionary contains `image` and (`label` or `labels`)
        """
        items = []
        for obj in self._datastore.objects:
            image = self._get_path(obj.image.id, full_path)
            if obj.labels:
                items.append({
                    "image": image,
                    "label": self._get_path(obj.labels.updated.id, full_path),
                })
                if obj.labels.original:
                    items[-1].update({
                        "original_label": self._get_path(obj.labels.original.id, full_path),
                    })
        return items

    def get_label_scores(self, image_id: str) -> List[ScoreModel]:

        image_id_exists = False

        for obj in self._datastore.objects:
            if image_id == obj.image.id and obj.labels is not None:
                return obj.labels.updated.scores

        if not image_id_exists:
            raise ImageNotFoundException(f"Image {image_id} not found")

    def set_label_score(self, image_id: str, score_name: str, score_value: Union[int, float]) -> None:

        image_id_exists = False

        for object_index, obj in enumerate(self._datastore.objects):
            if image_id == obj.image.id and obj.labels is not None:
                score_name_exists = False
                for score_index, score in enumerate(self._datastore.objects[object_index].labels.updated.scores):
                    if score_name == score.name:
                        score_name_exists = True
                        self._datastore.objects[object_index].labels.updated.scores[score_index].name = score_value

                if not score_name_exists:
                     self._datastore.objects[object_index].labels.updated.scores.append(
                         ScoreModel(name=score_name, value=score_value)
                     )


        if not image_id_exists:
            raise ImageNotFoundException(f"Image {image_id} not found")
