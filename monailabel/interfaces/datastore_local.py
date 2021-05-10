import io
import json
import os
import pathlib
import re
from uuid import uuid4

from openapi_schema_validator import validate

from monailabel.interfaces.datastore import Datastore


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

    `update_label_info(self, label_id: str, description: str) -> dict:`
        Update (or add if it does not exist) dataset label metadata as {id: description} map
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
            "info": {
                "type": "array",
                "minItems": 0,
                "properties": {
                    "id": {
                        "type": "string",
                        "nullable": False,
                        "pattern": "^[0-9a-zA-Z_-]{5,15}$",
                    },
                    "description": {
                        "type": "string",
                        "nullable": False,
                        "pattern": "^[0-9a-zA-Z_-]{5,256}$",
                    },
                },
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
                        "type": "object",
                        "minItems": 0,
                        "properties": {
                            "id": {
                                "type": "string",
                                "pattern": "^[0-9a-zA-Z_-]{5,15}",
                            },
                            "label": {
                                "type": "string",
                                "pattern": "^[0-9a-zA-Z_-]{2048}",
                            },
                        },
                    },
                },
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = None,
        dataset_config: str = "dataset.json",
    ):
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
            dataset_name = "new-dataset"

        # check if dataset configuration file exists
        if os.path.exists(self._dataset_config_path):
            with open(self._dataset_config_path) as f:
                self._dataset_config = json.load(f)
            validate(
                self._dataset_config["objects"],
                LocalDatastore._schema["properties"]["objects"],
            )
        else:
            files = LocalDatastore._list_files(dataset_path)
            self._dataset_config["name"] = dataset_name
            self._dataset_config.update(
                {
                    "objects": [
                        {
                            "image": file,
                            "labels": [],
                        }
                        for file in files
                    ],
                }
            )

            validate(self._dataset_config, LocalDatastore._schema)
            self._update_dataset()

    @property
    def name(self) -> str:
        """
        Dataset name (if one is assigned)

        Returns:
            name (str): Dataset name as string
        """
        return self._dataset_config.get("name")

    @name.setter
    def name(self, name: str):
        """
        Sets the dataset name in a standardized format (lowercase, no spaces).

            Parameters:
                name (str): Desired dataset name
        """
        standard_name = "".join(c.lower() if not c.isspace() else "-" for c in name)
        self._dataset_config.update({"name": standard_name})
        validate(self._dataset_config, LocalDatastore._schema)
        self._update_dataset()

    @property
    def description(self) -> str:
        """
        Gets the description field for the dataset
        """
        return self._dataset_config.get("description")

    @description.setter
    def description(self, description: str):
        """
        Set a description for the dataset
        """
        self._dataset_config.update({"description": description})
        validate(self._dataset_config, LocalDatastore._schema)
        self._update_dataset()

    def list_images(self) -> list:
        """
        List all the images in the dataset
        """
        return [obj["image"] for obj in self._dataset_config["objects"]]

    def list_labels(self) -> list:
        """
        List all the labels in the dataset
        """
        return [obj["labels"] for obj in self._dataset_config["objects"]]

    def find_objects(self, pattern: str, match_label: bool = False) -> list:
        """
        Find all the objects (image-[labels] pairings) that match the provided `pattern`

        Parameters:

        `pattern: str`
            a string to search the `image` fields of the dataset

        `match_label:bool`
            a boolean which, if set to true, will search for the same pattern in the `labels` field as well (default `False`)

        Returns:

        a list of matching objects in the form of a list of dictionaries of the form
        [{
            'image': '...',
            'labels': [
                '...label1',
                '...label2',
            ]
        },...]
        """
        if pattern is None:
            return self._dataset_config

        p = re.compile(pattern)
        matching_objects = []
        for obj in self._dataset_config["objects"]:
            if p.match(obj["image"]) or p.match(
                os.path.join(self._dataset_path, obj["image"])
            ):
                matching_objects.append(obj)
            if match_label and any(
                [
                    p.match(l) or p.match(os.path.join(self._dataset_path, l))
                    for l in obj["labels"]
                ]
            ):
                matching_objects.append(obj)
        return matching_objects

    def get_unlabeled_images(self) -> list:
        """
        Get a list of all the images without an associated label

        Returns:

            a list of image paths that do not have corresponding labels
        """
        images = []
        for obj in self._dataset_config["objects"]:
            if obj.get("labels") is None or len(obj.get("labels")) == 0:
                images.append(os.path.join(self._dataset_path, obj["image"]))
        return images

    def save_label(self, image: str, label_id: str, label: io.BytesIO) -> str:
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
        for i, obj in enumerate(self._dataset_config["objects"]):

            if image == obj["image"] or image == os.path.join(
                self._dataset_path, obj["image"]
            ):
                if label_id in obj["labels"]:
                    label_id = str(uuid4().hex) + "".join(
                        pathlib.Path(label_id).suffixes
                    )

                label_path = os.path.join(os.path.dirname(obj["image"]), label_id)
                self._dataset_config["objects"][i]["labels"].append(label_path)

                with open(os.path.join(self._dataset_path, label_path), "wb") as f:
                    label.seek(0)
                    f.write(label.getbuffer())

                break

        validate(self._dataset_config, LocalDatastore._schema)
        self._update_dataset()

        return label_path

    @property
    def info(self) -> dict:
        """
        The `info` section of the dataset, containing any label information if available
        """
        return self._dataset_config.get("info")

    def update_label_info(self, id: str, description: str) -> dict:
        """
        Update label info in the dataset
        """

        standard_id = "".join(c.lower() for c in id if not c.isspace())

        if (
            self._dataset_config.get("info") is not None
            and id in self._dataset_config["info"].keys()
        ):
            self._dataset_config["info"][id] = description

        else:
            self._dataset_config["info"].append({id: description})
        validate(self._dataset_config, LocalDatastore._schema)
        self._update_dataset()

        return {standard_id: self._dataset_config["info"][standard_id]}

    @staticmethod
    def _list_files(path: str):
        relative_file_paths = []
        for root, dirs, files in os.walk(path):
            base_dir = root.strip(path)
            relative_file_paths.extend([os.path.join(base_dir, file) for file in files])
        return relative_file_paths

    def _update_dataset(self):
        with open(self._dataset_config_path, "w") as f:
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
        for obj in self._dataset_config["objects"]:
            image = self._get_path(obj["image"], full_path)
            labels = obj.get("labels")
            if labels and len(labels):
                if flatten_labels:
                    for label in labels:
                        items.append(
                            {"image": image, "label": self._get_path(label, full_path)}
                        )
                else:
                    items.append(
                        {
                            "image": image,
                            "labels": [
                                self._get_path(label, full_path) for label in labels
                            ],
                        }
                    )
        return items
