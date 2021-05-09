import io
import json
import os
import pathlib
import re
from typing import List, Union, Dict
from uuid import uuid4


from openapi_schema_validator import validate

from monailabel.interfaces.datastore import Datastore, LabelStage
from monailabel.interfaces.exception import ImageNotFoundException, LabelNotFoundException


class LocalDatastore(Datastore):
    """
    Class to represent a datastore local to the MONAI-Label Server

    Attributes
    ----------
    `name: str`
        The name of the datastore

    `description: str`
        The description of the datastore

    `info: dict`
        A dictionary containing `{label_id: description}` map
    
    Methods:
    --------
    find_objects(self, pattern: str, match_label: bool=True) -> list:
        Finds objects with matching `pattern` allowing to search images (and labels as well if `match_label == True`)

    `get_unlabeled_images(self) -> list:`
        Get a list of image without any labels

    `list_images(self) -> list:`
        List all images

    `list_labels(self) -> list:`
        List all labels from all images

    `save_label(self, image: str, label_id: str, label: io.BytesIO):`
        Save a new label into the dataset with a given `label_id`.
        `label_id` saved with this method must match one of the `label_ids` in the dataset metadata
    """

    _schema = {
        "type": "object",
        "required": [
            "objects",
        ],
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[0-9a-zA-Z_-]{5,15}$",
            },
            "description": {
                "type": "string",
                "nullable": True,
                "pattern": "^[0-9a-zA-Z_-\s]{5,256}$",
            },
            "objects": {
                "type": "array",
                "minItems": 1,
                "properties": {
                    "image": {
                        "type": "string",
                        "pattern": "^\w{2048}",
                    },
                    "labels": {
                        "type": "array",
                        "minItems": "0",
                        "properties": {
                            "stage": {
                                "type": "string",
                                "pattern": "^[0-9a-zA-Z_-]{5,15}",
                                "nullable": False,
                            },
                            "id": {
                                "type": "string",
                                "pattern": "^[0-9a-zA-Z_-]{2048}",
                                "nullable": False,
                            },
                            "path": {
                                "type": "string",
                                "pattern": "^[0-9a-zA-Z_-]{4096}",
                                "nullable": False,
                            },
                            "scores": {
                                "type": "array",
                                "minItems": 0,
                                "properties": {
                                    "score_name": {
                                        "type": "string",
                                        "pattern": "^[0-9a-zA-Z_-]{5,15}",
                                    },
                                    "score_value": {
                                        "type": "number",
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
        "additionalProperties": False,
    }

    def __init__(self, dataset_path: str, dataset_name: str = None, dataset_config: str = 'dataset.json'):
        """
        Creates a `LocalDataset` object

        Parameters:

        `dataset_path: str`
            a string to the directory tree of the desired dataset

        `dataset_name: str`
            an optional name for the dataset

        `dataset_config: str`
            optional file name of the dataset configuration file (by default `dataset.json`)
        """
        self._dataset_path = dataset_path
        self._dataset_config_path = os.path.join(dataset_path, dataset_config)
        self._dataset_config = {}
        if dataset_name is None or len(dataset_config) == 0:
            dataset_name = 'new-dataset'

        # check if dataset configuration file exists
        if os.path.exists(self._dataset_config_path):
            with open(self._dataset_config_path) as f:
                self._dataset_config = json.load(f)
            validate(self._dataset_config['objects'], LocalDatastore._schema['properties']['objects'])
        else:
            files = LocalDatastore._list_files(dataset_path)
            self._dataset_config['name'] = dataset_name
            self._dataset_config.update({
                'objects': [{
                    'image': file,
                    'labels': [],
                } for file in files],
            })

            validate(self._dataset_config, LocalDatastore._schema)
            self._update_datastore_file()

    @property
    def name(self) -> str:
        """
        Dataset name (if one is assigned)

        Returns:
            name (str): Dataset name as string
        """
        return self._dataset_config.get('name')

    @name.setter
    def name(self, name: str):
        """
        Sets the dataset name in a standardized format (lowercase, no spaces).

            Parameters:
                name (str): Desired dataset name
        """
        standard_name = ''.join(c.lower() if not c.isspace() else '-' for c in name)
        self._dataset_config.update({'name': standard_name})
        validate(self._dataset_config, LocalDatastore._schema)
        self._update_datastore_file()

    @property
    def description(self) -> str:
        """
        Gets the description field for the dataset
        """
        return self._dataset_config.get('description')

    @description.setter
    def description(self, description: str):
        """
        Set a description for the dataset
        """
        self._dataset_config.update({'description': description})
        validate(self._dataset_config, LocalDatastore._schema)
        self._update_datastore_file()

    def list_images(self) -> list:
        """
        List all the images in the dataset
        """
        return [obj['image'] for obj in self._dataset_config['objects']]

    def list_labels(self) -> list:
        """
        List all the labels in the dataset
        """
        return [obj['labels'] for obj in self._dataset_config['objects']]

    def find_data_by_image(self, pattern: str) -> list:
        """
        Find all the objects (image-[labels] pairings) that match the provided `pattern`

        Parameters:

        `pattern: str`
            a string to search the `image` fields of the dataset

        Returns:

        a list of matching objects in the form of a list of dictionaries
        """
        if pattern is None:
            return self._dataset_config

        p = re.compile(pattern)
        matching_objects = []
        for obj in self._dataset_config['objects']:
            if p.match(obj['image']) or p.match(os.path.join(self._dataset_path, obj['image'])):
                matching_objects.append(obj)
        return matching_objects

    def find_data_by_label(self, pattern: str) -> list:
        """
        Find all the objects (image-[labels] pairings) that match the provided `pattern`

        Parameters:

        `pattern: str`
            a string to search the `image` fields of the dataset

        Returns:

        a list of matching objects in the form of a list of dictionaries
        """
        if pattern is None:
            return self._dataset_config

        p = re.compile(pattern)
        matching_objects = []
        for obj in self._dataset_config['objects']:
            if any([p.match(l['id']) for l in obj['labels']]):
                matching_objects.append(obj)
        return matching_objects

    def get_unlabeled_images(self) -> list:
        """
        Get a list of all the images without an associated label

        Returns:

            a list of image paths that do not have corresponding labels
        """
        images = []
        for obj in self._dataset_config['objects']:
            if obj.get('labels') is None or len(obj.get('labels')) == 0:
                images.append(os.path.join(self._dataset_path, obj['image']))
        return images

    def save_label(self, image: str, label_stage: LabelStage, label_id: str, label: io.BytesIO, scores: List[Dict[str, Union[int, float]]]=[]) -> str:
        """
        Save the label for the given image in the dataset

        Parameters:

        `image: str`
            the path of the image file in the dataset to which the provided `label_id` and `label` should be associated

        `label_id`
            the file name of the label (should contain file extension e.g. `.nii`); if the provided file name already exists it
            will be updated

        `label: io.BytesIO`
            a byte stream of the file to be saved in the dataset

        Returns:

        the updated `label_id` (if there is a clash or the original `label_id` provided if not) prepended by any relative path
        to the `image` path provided in the parameters
        """

        label_path = None
        image_exists = False

        for i, obj in enumerate(self._dataset_config['objects']):

            if image == obj['image'] or image == os.path.join(self._dataset_path, obj['image']):

                image_exists = True

                if label_id in obj['labels']:
                    label_id = str(uuid4().hex) + ''.join(pathlib.Path(label_id).suffixes)

                label_path = os.path.join(os.path.dirname(obj['image']), label_id)

                # if the user is adding a label to save the keep track of it and add it as a `SAVED` label
                if label_stage == LabelStage.SAVED:

                    num_labels = len(self._dataset_config['objects'][i]['labels'])
                    updated_label_stage = f'{LabelStage.SAVED}-{num_labels:06}'

                    self._dataset_config['objects'][i]['labels'].append({
                        "stage" : updated_label_stage,
                        "id": label_id,
                        "path": label_path,
                        "scores": scores,
                    })

                # the user is adding an initial label to compare against and generate a score later on
                # the original label is overwritten; one may not have more than one ORIGINAL labels
                elif label_stage == LabelStage.ORIGINAL:

                    self._dataset_config['objects'][i]['labels'] = {
                        "stage": LabelStage.ORIGINAL,
                        "id": label_path,
                        "scores": [],
                    }

                with open(os.path.join(self._dataset_path, label_path), 'wb') as f:
                    label.seek(0)
                    f.write(label.getbuffer())

                break

        if not image_exists:
            raise ImageNotFoundException(f"Image {image} not found")

        validate(self._dataset_config, LocalDatastore._schema)
        self._update_datastore_file()

        return label_path

    @staticmethod
    def _list_files(path: str):
        relative_file_paths = []
        for root, dirs, files in os.walk(path):
            base_dir = root.strip(path)
            relative_file_paths.extend([os.path.join(base_dir, file) for file in files])
        return relative_file_paths

    def _update_datastore_file(self):
        with open(self._dataset_config_path, 'w') as f:
            json.dump(self._dataset_config, f, indent=2)

    def _get_path(self, path: str, full_path=True):
        if not full_path or os.path.isabs(path):
            return path
        return os.path.realpath(os.path.join(self._dataset_path, path))

    def datalist(self, full_path=True, flatten_labels=True) -> list:
        """
        Get Data List of image and valid/existing label pairs

        :param full_path: Add full path for image/label objects
        :param flatten_labels: Flatten labels
        :return: List of dictionary objects.  Each dictionary contains `image` and (`label` or `labels`)
        """
        items = []
        for obj in self._dataset_config['objects']:
            image = self._get_path(obj["image"], full_path)
            labels = obj.get('labels')
            if labels and len(labels):
                if flatten_labels:
                    for label in labels:
                        items.append({
                            "image": image,
                            "label": self._get_path(label['path'], full_path)
                        })
                else:
                    items.append({
                        "image": image,
                        "labels": [self._get_path(label['path'], full_path) for label in labels]
                    })
        return items

    def get_label_scores(self, label_id: str) -> List[Dict[str, Union[int, float]]]:

        for obj in self._dataset_config['objects']:
            labels = obj.get('labels')
            if labels and len(labels):
                for label in labels:
                    if label['id'] == label_id:
                        return label['scores']

        raise LabelNotFoundException(f"Label {label_id} not found")

    def set_label_score(self, label_id: str, score_name: str, score_value: Union[int, float]) -> None:

        label_id_exists = False

        for obj in self._dataset_config['objects']:
            labels = obj.get('labels')
            if labels and len(labels):
                for i, label in enumerate(labels):
                    if label['id'] == label_id:
                        label_id_exists = True
                        labels[i]['scores'].update({
                            "score_name": score_name,
                            "score_value": score_value
                        })

        if not label_id_exists:
            raise LabelNotFoundException(f"Label {label_id} not found")
