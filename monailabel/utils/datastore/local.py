import io
import json
import os
import pathlib
import shutil
from typing import Any, Dict, List

from pydantic import BaseModel

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
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

    def __init__(
        self,
        datastore_path: str,
        datastore_config: str = "datastore.json",
        label_store_path: str = "labels",
        reconcile_datastore: bool = True,
    ):
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
        self._label_store_path = label_store_path

        if os.path.exists(self._datastore_config_path):
            # check if dataset configuration file exists and load if it does
            self._datastore = LocalDatastoreModel.parse_file(self._datastore_config_path)
        else:
            # otherwise create anew
            self._datastore = LocalDatastoreModel(name="new-dataset", description="New Dataset")

        # ensure labels path exists regardless of whether a datastore file is present
        os.makedirs(os.path.join(self._datastore_path, self._label_store_path), exist_ok=True)

        # reconcile the loaded datastore file with any existing files in the path
        if reconcile_datastore:
            self._reconcile_datastore()

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

    @property
    def datastore_path(self):
        return self._datastore_path

    @property
    def labelstore_path(self):
        return os.path.join(self._datastore_path, self._label_store_path)

    def datalist(self, full_path=True) -> List[Dict[str, str]]:
        """
        Return a dictionary of image and label pairs corresponding to the 'image' and 'label'
        keys respectively

        :return: the {'label': image, 'label': label} pairs for training
        """
        items = []
        for obj in self._datastore.objects:
            image_path = self._get_path(obj.image.id, False, full_path)
            for label in obj.labels:
                if label.tag == DefaultLabelTag.FINAL.value:
                    items.append(
                        {
                            "image": image_path,
                            "label": self._get_path(label.id, True, full_path),
                        }
                    )
        return items

    def get_image(self, image_id: str) -> io.BytesIO:
        """
        Retrieve image object based on image id

        :param image_id: the desired image's id
        :return: return the "image"
        """
        buf = None
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                with open(os.path.join(self._datastore_path, obj.image.id), "rb") as f:
                    buf = io.BytesIO(f.read())
                break
        return buf

    def get_image_uri(self, image_id: str) -> str:
        """
        Retrieve image uri based on image id

        :param image_id: the desired image's id
        :return: return the image uri
        """
        image_path = None
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                image_path = os.path.join(self._datastore_path, obj.image.id)
                break
        return image_path

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        """
        Get the image information for the given image id

        :param image_id: the desired image id
        :return: image info as a list of dictionaries Dict[str, Any]
        """
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                return obj.image.info

        return {}

    def get_label(self, label_id: str) -> io.BytesIO:
        """
        Retrieve image object based on label id

        :param label_id: the desired label's id
        :return: return the "label"
        """
        buf = None
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    with open(os.path.join(self._datastore_path, self._label_store_path, label.id)) as f:
                        buf = io.BytesIO(f.read())
        return buf

    def get_label_uri(self, label_id: str) -> str:
        """
        Retrieve label uri based on image id

        :param label_id: the desired label's id
        :return: return the label uri
        """
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    return os.path.join(self._datastore_path, self._label_store_path, label.id)
        return None

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        """
        Retrieve all label ids for the given image id

        :param image_id: the desired image's id
        :return: label ids mapped to the appropriate tag as Dict[str, str]
        """
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                labels = {label.id: label.tag for label in obj.labels}
                return labels
        return {}

    def get_label_info(self, label_id: str) -> Dict[str, Any]:
        """
        Get the label information for the given label id

        :param label_id: the desired label id
        :return: label info as a list of dictionaries Dict[str, Any]
        """
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    return label.info

        return {}

    def get_labeled_images(self) -> List[str]:
        """
        Get all images that have a corresponding label

        :return: list of image ids List[str]
        """
        image_ids = []
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.tag == DefaultLabelTag.FINAL.value:
                    image_ids.append(obj.image.id)

        return image_ids

    def get_unlabeled_images(self) -> List[str]:
        """
        Get all images that have no corresponding label

        :return: list of image ids List[str]
        """
        image_ids = []
        for obj in self._datastore.objects:
            if not obj.labels or DefaultLabelTag.FINAL.value not in [label.tag for label in obj.labels]:
                image_ids.append(obj.image.id)

        return image_ids

    def list_images(self) -> List[str]:
        """
        Return list of image ids available in the datastore

        :return: list of image ids List[str]
        """
        return [obj.image.id for obj in self._datastore.objects]

    def refresh(self):
        """
        Refresh the datastore based on the state of the files on disk
        """
        self._reconcile_datastore()
        self._update_datastore_file()

    def save_label(self, image_id: str, label_filename: str, label_tag: str) -> str:
        """
        Save a label for the given image id and return the newly saved label's id

        :param image_id: the image id for the label
        :param label_filename: the path to the label file
        :param label_tag: the tag for the label
        :return: the label id for the given label filename
        """
        for obj in self._datastore.objects:

            if obj.image.id == image_id:

                image_ext = "".join(pathlib.Path(image_id).suffixes)
                label_ext = "".join(pathlib.Path(label_filename).suffixes)
                label_id = "label_" + label_tag + "_" + image_id.replace(image_ext, "") + label_ext

                datastore_label_path = os.path.join(self._datastore_path, self._label_store_path, label_id)
                shutil.copy(src=label_filename, dst=datastore_label_path, follow_symlinks=True)

                if label_tag not in [label.tag for label in obj.labels]:
                    obj.labels.append(
                        LabelModel(
                            id=label_id,
                            tag=label_tag,
                        )
                    )
                else:
                    for label_index, label in enumerate(obj.labels):
                        if label.tag == label_tag:
                            obj.labels[label_index] = LabelModel(id=label_id, tag=label_tag)

                self._update_datastore_file()

                return label_id

        else:
            raise ImageNotFoundException(f"Image {image_id} not found")

    def update_image_info(self, image_id: str, info: Dict[str, Any]) -> None:
        """
        Update (or create a new) info tag for the desired image

        :param image_id: the id of the image we want to add/update info
        :param info: a dictionary of custom image information Dict[str, Any]
        """
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                obj.image.info.update(info)
                self._update_datastore_file()
                break
        else:
            raise ImageNotFoundException(f"Image {image_id} not found")

    def update_label_info(self, label_id: str, info: Dict[str, Any]) -> None:
        """
        Update (or create a new) info tag for the desired label

        :param label_id: the id of the label we want to add/update info
        :param info: a dictionary of custom label information Dict[str, Any]
        """
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    label.info.update(info)
                    self._update_datastore_file()
                return

        raise LabelNotFoundException(f"Label {label_id} not found")

    def _get_path(self, path: str, is_label: bool, full_path=True):
        if is_label:
            path = os.path.join(self._label_store_path, path)

        if not full_path or os.path.isabs(path):
            return path

        return os.path.realpath(os.path.join(self._datastore_path, path))

    @staticmethod
    def _list_files(path: str):
        relative_file_paths = []
        for root, dirs, files in os.walk(path):
            base_dir = root.lstrip(path)
            relative_file_paths.extend([os.path.join(base_dir, file) for file in files])
        return relative_file_paths

    def _reconcile_datastore(self):
        self._remove_object_with_missing_file()
        self._add_object_with_present_file()

    def _remove_object_with_missing_file(self) -> None:
        """
        remove objects present in the datastore file but not present on path
        (even if labels exist, if images do not the whole object is removed from the datastore)
        """
        files = LocalDatastore._list_files(self._datastore_path)
        image_id_files = [file for file in files if not file.startswith(self._label_store_path)]
        image_id_datastore = [obj.image.id for obj in self._datastore.objects]
        missing_file_image_id = list(set(image_id_datastore) - set(image_id_files))
        if missing_file_image_id:
            self._datastore.objects = [
                obj for obj in self._datastore.objects if obj.image.id not in missing_file_image_id
            ]

        label_id_files = [pathlib.Path(file).name for file in files if file.startswith(self._label_store_path)]
        label_id_datastore = [label.id for obj in self._datastore.objects for label in obj.labels]

        missing_file_label_id = list(set(label_id_datastore) - set(label_id_files))
        if missing_file_label_id:
            for obj in self._datastore.objects:
                obj.labels = [label for label in obj.labels if label.id not in missing_file_label_id]

    def _add_object_with_present_file(self) -> None:
        """
        add objects which are not present in the datastore file, but are present in the datastore directory
        this adds the image present in the datastore path and any corresponding labels for that image
        """
        files = LocalDatastore._list_files(self._datastore_path)
        image_id_files = [
            file
            for file in files
            if not file.startswith(self._label_store_path) and file != pathlib.Path(self._datastore_config_path).name
        ]

        # add any missing image files and any corresponding labels
        existing_image_ids = [obj.image.id for obj in self._datastore.objects]
        for image_id in image_id_files:

            image_ext = "".join(pathlib.Path(image_id).suffixes)
            image_id_nosuffix = image_id.replace(image_ext, "")

            # add the image i if not present
            if image_id not in existing_image_ids:
                self._datastore.objects.append(ObjectModel(image=ImageModel(id=image_id)))

            # find label files related to the image id being added to the datastore
            label_id_files = [
                pathlib.Path(file).name
                for file in files
                if file.startswith(self._label_store_path) and "label_" in file and image_id_nosuffix in file
            ]
            for label_id in label_id_files:
                image_id_index = [obj.image.id for obj in self._datastore.objects].index(image_id)
                label_parts = label_id.split(image_id_nosuffix)
                label_tag = label_parts[0].replace("label_", "").strip("_")
                if label_id not in [label.id for label in self._datastore.objects[image_id_index].labels]:
                    self._datastore.objects[image_id_index].labels.append(LabelModel(id=label_id, tag=label_tag))

    def _update_datastore_file(self):
        with open(self._datastore_config_path, "w") as f:
            json.dump(self._datastore.dict(), f, indent=2, default=str)
