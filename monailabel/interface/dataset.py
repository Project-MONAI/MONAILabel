from abc import ABCMeta, abstractmethod
import io
import os
import re

import json

from openapi_schema_validator import validate


class Dataset(metaclass=ABCMeta):

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
    def save_label(self, image: str, label_id: str, label: io.BytesIO): pass

    @abstractmethod
    def get_unlabeled_images(self) -> list: pass


class LocalDataset(Dataset):
    """
    Class to represent a dataset local to the MONAI-Label Server

    Attributes
    ----------
    `name: str`
        The name of the dataset

    `description: str`
        The description of the dataset

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
                "pattern": "^[^-][\w{5-15}-]$",
            },
            "description": {
                "type": "string",
                "pattern": "^\w{5-256}$",
            },
            "info": {
                "type": "array",
                "minItems": 0,
                "properties": {
                    "id": {
                        "type": "string",
                        "pattern": "^\w{5-15}$",
                    },
                    "description": {
                        "type": "string",
                        "pattern": "^\w{5-256}$",
                    },
                }
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
                            "index": {
                                "type": "integer",
                                "pattern": "^\w{5-15}",
                            },
                            "label": {
                                "type": "string",
                                "pattern": "^\w{2048}",
                            },
                        }
                    }
                }
            }
        },
        "additionalProperties": False,
    }

    def __init__(self, dataset_path: str, dataset_name: str=None, dataset_config: str='dataset.json'):
        self._dataset_path = dataset_path
        self._dataset_config_path = os.path.join(dataset_path, dataset_config)
        self._dataset_config = None
        
        # check if dataset configuration file exists
        if os.path.exists(self._dataset_config_path):
            with open(self._dataset_config_path) as f:
                self._dataset_config = json.load(f)
            validate(self._dataset_config['objects'], LocalDataset._schema['properties']['objects'])
        else:
            files = LocalDataset._list_files(dataset_path)
            self._dataset_config.name = dataset_name
            self._dataset_config = {
                    'objects': [{
                        'image': file,
                        'labels': [],
                    } for file in files],
                }

            validate(self._dataset_config['objects'], LocalDataset._schema['objects'])
            self._update_dataset()

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
        validate(self._dataset_config['name'], LocalDataset._schema['properties']['name'])
        self._update_dataset()

    @property
    def description(self) -> str:
        return self._dataset_config.get('description')

    @description.setter
    def description(self, description: str):
        self._dataset_config.update({'description': description})
        validate(self._dataset_config, LocalDataset._schema)
        self._update_dataset()

    def list_images(self) -> list:
        return [obj['image'] for obj in self._dataset_config['objects']]

    def list_labels(self) -> list:
        return [obj['labels'] for obj in self._dataset_config['objects']]

    def find_objects(self, pattern: str, match_label: bool=False) -> list:
        p = re.compile(pattern)
        matching_objects = []
        for obj in self._dataset_config['objects']:
            if p.match(obj['image']):
                matching_objects.append(obj)
            if match_label and any([p.match(l) for l in obj['labels']]):
                matching_objects.append(obj)
        return matching_objects

    def get_unlabeled_images(self) -> list:
        images = []
        for obj in self._dataset_config['objects']:
            if obj.get('labels') is None or len(obj.get('labels')) == 0:
                images.append(obj['image'])
        return images
    
    def save_label(self, image: str, label_name: str, label: io.BytesIO):
        
        for obj in self._dataset_config['objects']:
        
            if image == self._dataset_config['image']:
                label_path = self._dataset_config['image']
                self._dataset_config['labels'].append(os.path.join(os.path.basename(self._dataset_config['image']), label_name))

                with open(self._dataset_config['label'][-1], 'wb') as f:
                    label.seek(0)
                    f.write(label.getbuffer())
        
        validate(self._dataset_config, LocalDataset._schema)
        self._update_dataset()
    
    @property
    def info(self) -> dict:
        return self._dataset_config.get('info')

    def update_label_info(self, id: str, description: str) -> dict:

        standard_id = ''.join(c.lower() for c in id if not c.isspace())
        
        if self._dataset_config.get('info') is not None and id in self._dataset_config['info'].keys():
            self._dataset_config['info'][id] = description
        
        else:
            self._dataset_config['info'].append({id: description})
        validate(self._dataset_config, LocalDataset._schema)
        self._update_dataset()

        return {standard_id: _dataset_config['info'][standard_id]}

    @staticmethod
    def _list_files(path: str):
        relative_file_paths = []
        for root, dirs, files in os.walk(path):
            base_dir = root.strip(path)
            relative_file_paths.extend([os.path.join(base_dir, file) for file in files])
        return relative_file_paths

    def _update_dataset(self):
        with open(self._dataset_config_path, 'w') as f:
                json.dump(self._dataset_config, f)

if __name__ == "__main__":
    ds = LocalDataset('/raid/datasets/Task09_Spleen/imagesTs')
    ds.name = 'test dataset'
    ds.description = 'my description'
    ds.update_info('liver', 'Liver organ tissue')
    ds.update_info('tumor', 'Liver tumor tissue')