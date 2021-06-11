import fnmatch
import io
import json
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Dict, List

from filelock import FileLock
from pydantic import BaseModel
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.exception import ImageNotFoundException, LabelNotFoundException

logger = logging.getLogger(__name__)


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
        image_extensions=("*.nii.gz", "*.nii"),
        label_extensions=("*.nii.gz", "*.nii"),
        auto_reload=False,
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
        self._image_extensions = [image_extensions] if isinstance(image_extensions, str) else image_extensions
        self._label_extensions = [label_extensions] if isinstance(label_extensions, str) else label_extensions
        self._ignore_event_count = 0
        self._ignore_event_config = False
        self._config_ts = 0
        self._auto_reload = auto_reload

        self._lock = FileLock(os.path.join(datastore_path, ".lock"))
        logging.getLogger("filelock").setLevel(logging.ERROR)

        logger.info(f"Image Extensions: {self._image_extensions}")
        logger.info(f"Label Extensions: {self._label_extensions}")
        logger.info(f"Auto Reload: {auto_reload}")

        self._datastore: LocalDatastoreModel = LocalDatastoreModel(name="new-dataset", description="New Dataset")
        self._init_from_datastore_file(throw_exception=True)

        # ensure labels path exists regardless of whether a datastore file is present
        os.makedirs(os.path.join(self._datastore_path, self._label_store_path), exist_ok=True)

        # reconcile the loaded datastore file with any existing files in the path
        self._reconcile_datastore()

        if auto_reload:
            logger.info("Start observing external modifications on datastore (AUTO RELOAD)")
            include_patterns = [
                f"{self._datastore_path}{os.path.sep}{ext}" for ext in [*image_extensions, *label_extensions]
            ]
            include_patterns.extend(
                [
                    f"{os.path.join(self._datastore_path, self._label_store_path)}{os.path.sep}{ext}"
                    for ext in [*image_extensions, *label_extensions]
                ]
            )
            include_patterns.append(self._datastore_config_path)

            self._handler = PatternMatchingEventHandler(patterns=include_patterns)
            self._handler.on_created = self._on_any_event
            self._handler.on_deleted = self._on_any_event
            self._handler.on_modified = self._on_modify_event

            try:
                self._ignore_event_count = 0
                self._ignore_event_config = False
                self._observer = Observer()
                self._observer.schedule(self._handler, recursive=True, path=self._datastore_path)
                self._observer.start()
            except OSError as e:
                logger.error(
                    "Failed to start File watcher. "
                    "Local datastore will not update if images and labels are moved from datastore location."
                )
                logger.error(str(e))

    def name(self) -> str:
        """
        Dataset name (if one is assigned)

        Returns:
            name (str): Dataset name as string
        """
        return self._datastore.name

    def set_name(self, name: str):
        """
        Sets the dataset name in a standardized format (lowercase, no spaces).

            Parameters:
                name (str): Desired dataset name
        """
        self._datastore.name = name
        self._update_datastore_file()

    def description(self) -> str:
        """
        Gets the description field for the dataset

        :return description: str
        """
        return self._datastore.description

    def set_description(self, description: str):
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
                if label.tag == DefaultLabelTag.FINAL:
                    items.append(
                        {
                            "image": image_path,
                            "label": self._get_path(label.id, True, full_path),
                        }
                    )
        return items

    def get_image(self, image_id: str) -> Any:
        """
        Retrieve image object based on image id

        :param image_id: the desired image's id
        :return: return the "image"
        """
        buf = None
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                buf = io.BytesIO(pathlib.Path(os.path.join(self._datastore_path, obj.image.id)).read_bytes())
                break
        return buf

    def get_image_uri(self, image_id: str) -> str:
        """
        Retrieve image uri based on image id

        :param image_id: the desired image's id
        :return: return the image uri
        """
        image_path = ""
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

    def get_label(self, label_id: str) -> Any:
        """
        Retrieve image object based on label id

        :param label_id: the desired label's id
        :return: return the "label"
        """
        buf = None
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.id == label_id:
                    buf = io.BytesIO(
                        pathlib.Path(os.path.join(self._datastore_path, self._label_store_path, label.id)).read_bytes()
                    )
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
        return ""

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        """
        Retrieve all label ids for the given image id

        :param image_id: the desired image's id
        :return: label ids mapped to the appropriate `LabelTag` as Dict[str, LabelTag]
        """
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                return {label.id: label.tag for label in obj.labels}
        return {}

    def get_label_by_image_id(self, image_id: str, tag: str) -> str:
        """
        Retrieve label id for the given image id and tag

        :param image_id: the desired image's id
        :param tag: matching tag name
        :return: label id
        """
        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                for label in obj.labels:
                    if label.tag == tag:
                        return label.id
        return ""

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
                if label.tag == DefaultLabelTag.FINAL:
                    image_ids.append(obj.image.id)

        return image_ids

    def get_unlabeled_images(self) -> List[str]:
        """
        Get all images that have no corresponding label

        :return: list of image ids List[str]
        """
        image_ids = []
        for obj in self._datastore.objects:
            if not obj.labels or DefaultLabelTag.FINAL not in [label.tag for label in obj.labels]:
                image_ids.append(obj.image.id)

        return image_ids

    def list_images(self) -> List[str]:
        """
        Return list of image ids available in the datastore

        :return: list of image ids List[str]
        """
        return [obj.image.id for obj in self._datastore.objects]

    def _on_any_event(self, event):
        if self._ignore_event_count:
            logger.debug(f"Ignoring event by count: {self._ignore_event_count} => {event}")
            self._ignore_event_count = max(self._ignore_event_count - 1, 0)
            return

        logger.debug(f"Event: {event}")
        self.refresh()

    def _on_modify_event(self, event):
        # handle modify events only for config path; rest ignored
        if event.src_path != self._datastore_config_path:
            return

        if self._ignore_event_config:
            self._ignore_event_config = False
            return

        self._init_from_datastore_file()

    def refresh(self):
        """
        Refresh the datastore based on the state of the files on disk
        """
        self._init_from_datastore_file()
        self._reconcile_datastore()

    def add_image(self, image_id: str, image_filename: str) -> str:
        if not image_id:
            image_id = os.path.basename(image_filename)

        logger.info(f"Adding Image: {image_id}")
        shutil.copy(image_filename, os.path.join(self._datastore_path, image_id))
        if not self._auto_reload:
            self.refresh()
        return image_id

    def remove_image(self, image_id: str) -> None:
        logger.info(f"Removing Image: {image_id}")

        # Remove all labels
        label_ids = self.get_labels_by_image_id(image_id)
        for label_id in label_ids:
            self.remove_label(label_id)

        # Remove Image
        p = os.path.join(self._datastore_path, image_id)
        if os.path.exists(p):
            os.unlink(p)

        if not self._auto_reload:
            self.refresh()

    def save_label(self, image_id: str, label_filename: str, label_tag: str) -> str:
        """
        Save a label for the given image id and return the newly saved label's id

        :param image_id: the image id for the label
        :param label_filename: the path to the label file
        :param label_tag: the tag for the label
        :return: the label id for the given label filename
        """
        logger.info(f"Saving Label for Image: {image_id}; Tag: {label_tag}")

        for obj in self._datastore.objects:
            if obj.image.id == image_id:
                image_ext = "".join(pathlib.Path(image_id).suffixes)
                label_ext = "".join(pathlib.Path(label_filename).suffixes)
                label_id = "label_" + label_tag + "_" + image_id.replace(image_ext, "") + label_ext

                with self._lock:
                    os.makedirs(os.path.join(self._datastore_path, self._label_store_path), exist_ok=True)
                    datastore_label_path = os.path.join(self._datastore_path, self._label_store_path, label_id)
                    shutil.copy(src=label_filename, dst=datastore_label_path, follow_symlinks=True)

                    lm = LabelModel(id=label_id, tag=label_tag, info={"ts": int(time.time())})
                    if label_tag not in [label.tag for label in obj.labels]:
                        obj.labels.append(lm)
                    else:
                        for label_index, label in enumerate(obj.labels):
                            if label.tag == label_tag:
                                obj.labels[label_index] = lm  # This will reset the previous info

                    self._update_datastore_file(lock=False)
                return label_id

        raise ImageNotFoundException(f"Image {image_id} not found")

    def remove_label(self, label_id: str) -> None:
        logger.info(f"Removing label: {label_id}")
        p = os.path.join(self.labelstore_path, label_id)
        if os.path.exists(p):
            os.unlink(p)
        if not self._auto_reload:
            self.refresh()

    def remove_label_by_tag(self, label_tag: str) -> None:
        label_ids = []
        for obj in self._datastore.objects:
            for label in obj.labels:
                if label.tag == label_tag:
                    label_ids.append(label.id)

        logger.info(f"Tag: {label_tag}; Removing label(s): {label_ids}")
        for label_id in label_ids:
            self.remove_label(label_id)

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
                return

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
    def _list_files(path, patterns):
        files = os.listdir(path)

        filtered = dict()
        for pattern in patterns:
            matching = fnmatch.filter(files, pattern)
            for file in matching:
                filtered[os.path.basename(file)] = file
        return filtered

    def _reconcile_datastore(self):
        if self._remove_object_with_missing_file() or self._add_object_with_present_file():
            self._update_datastore_file()

    def _remove_object_with_missing_file(self) -> bool:
        """
        remove objects present in the datastore file but not present on path
        (even if labels exist, if images do not the whole object is removed from the datastore)
        """
        invalidate = False

        image_id_files = self._list_files(self._datastore_path, self._image_extensions)
        image_id_datastore = [obj.image.id for obj in self._datastore.objects]
        missing_file_image_id = list(set(image_id_datastore) - set(image_id_files.keys()))
        if missing_file_image_id:
            logger.info(f"Removing Missing Images: {missing_file_image_id}")
            invalidate = True
            self._datastore.objects = [
                obj for obj in self._datastore.objects if obj.image.id not in missing_file_image_id
            ]

        label_id_files = self._list_files(
            os.path.join(self._datastore_path, self._label_store_path), self._label_extensions
        )
        label_id_datastore = [label.id for obj in self._datastore.objects for label in obj.labels]
        missing_file_label_id = list(set(label_id_datastore) - set(label_id_files.keys()))
        if missing_file_label_id:
            logger.info(f"Removing Missing Labels: {missing_file_label_id}")
            invalidate = True
            for obj in self._datastore.objects:
                obj.labels = [label for label in obj.labels if label.id not in missing_file_label_id]

        return invalidate

    def _add_object_with_present_file(self) -> bool:
        """
        add objects which are not present in the datastore file, but are present in the datastore directory
        this adds the image present in the datastore path and any corresponding labels for that image
        """
        invalidate = False

        image_id_files = LocalDatastore._list_files(self._datastore_path, self._image_extensions)
        label_id_files = LocalDatastore._list_files(
            os.path.join(self._datastore_path, self._label_store_path), self._label_extensions
        )

        # add any missing image files and any corresponding labels
        existing_image_ids = [obj.image.id for obj in self._datastore.objects]
        for image_id in image_id_files:

            image_ext = "".join(pathlib.Path(image_id).suffixes)
            image_id_nosuffix = image_id.replace(image_ext, "")

            # add the image i if not present
            if image_id not in existing_image_ids:
                logger.info(f"Adding New Image: {image_id}")
                invalidate = True
                self._datastore.objects.append(ObjectModel(image=ImageModel(id=image_id)))

            # for matching image ids only
            for label_id in [l_id for l_id in label_id_files if f"{image_id_nosuffix}." in l_id]:
                image_id_index = [obj.image.id for obj in self._datastore.objects].index(image_id)
                label_parts = label_id.split(image_id_nosuffix)
                label_tag = label_parts[0].replace("label_", "").strip("_")

                if label_id not in [label.id for label in self._datastore.objects[image_id_index].labels]:
                    logger.info(f"Adding New Label: {image_id} => {label_id}")
                    invalidate = True
                    self._datastore.objects[image_id_index].labels.append(LabelModel(id=label_id, tag=label_tag))

        return invalidate

    def _init_from_datastore_file(self, throw_exception=False):
        try:
            with self._lock:
                if os.path.exists(self._datastore_config_path):
                    ts = os.stat(self._datastore_config_path).st_mtime
                    if self._config_ts != ts:
                        logger.debug(f"Reload Datastore; old ts: {self._config_ts}; new ts: {ts}")
                        self._datastore = LocalDatastoreModel.parse_file(self._datastore_config_path)
                        self._config_ts = ts
        except ValueError as e:
            logger.error(f"+++ Failed to load datastore => {e}")
            if throw_exception:
                raise e

    def _update_datastore_file(self, lock=True):
        if lock:
            self._lock.acquire()

        logger.debug("+++ Datastore is updated...")
        self._ignore_event_config = True
        with open(self._datastore_config_path, "w") as f:
            f.write(json.dumps(self._datastore.dict(), indent=2, default=str))
        self._config_ts = os.stat(self._datastore_config_path).st_mtime

        if lock:
            self._lock.release()

    def status(self) -> Dict[str, Any]:
        tags: dict = {}
        for obj in self._datastore.objects:
            for label in obj.labels:
                tags[label.tag] = tags.get(label.tag, 0) + 1

        return {
            "total": len(self.list_images()),
            "completed": len(self.get_labeled_images()),
            "label_tags": tags,
            "train": self.datalist(full_path=False),
        }

    def __str__(self):
        return json.dumps(self._datastore.dict())
