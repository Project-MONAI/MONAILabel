import hashlib
import io
import json
import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Dict, List

from filelock import FileLock

from monailabel.interfaces import Datastore
from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.utils.datastore.dicom.client import DICOMWebClient
from monailabel.utils.datastore.dicom.convert import ConverterUtil
from monailabel.utils.datastore.dicom.datamodel import DICOMWebDatastoreModel
from monailabel.utils.others.generic import file_checksum

logger = logging.getLogger(__name__)


class DICOMWebCache(Datastore):
    """
    Class to represent a DICOMWeb datastore for the MONAI-Label Server

    Attributes
    ----------
    `name: str`
        The name of the datastore

    `description: str`
        The description of the datastore
    """

    def __init__(
        self,
        dicomweb_client: DICOMWebClient,
        label_store_path: str = "labels",
        datastore_config: str = "datastore.json",
    ):
        self._config_ts = 0
        self._dicomweb_client = dicomweb_client
        dicomweb_uri = self._dicomweb_client.base_url
        self._datastore_path = os.path.join(
            Path.home(),
            '.cache',
            'monailabel',
            hashlib.md5(dicomweb_uri.encode('utf-8')).hexdigest(),
        )
        self._lock = FileLock(os.path.join(self._datastore_path, ".lock"))
        self._datastore_config_path = os.path.join(self._datastore_path, datastore_config)
        self._label_path = label_store_path

        os.makedirs(self._datastore_path, exist_ok=True)
        os.makedirs(os.path.join(self._datastore_path, self._label_path), exist_ok=True)

        logger.info(f"DICOMWeb Endpoint: {dicomweb_uri}")
        logger.info(f"Datastore cache path: {self._datastore_path}")

        self._datastore: DICOMWebDatastoreModel = DICOMWebDatastoreModel(
            url=f"{dicomweb_uri}", description="Local Cache for DICOMWeb")

        self._datastore.objects = self._dicomweb_client.retrieve_dataset()
        self._update_datastore_file()

    def name(self) -> str:
        return self._datastore.url

    def set_name(self, name: str):
        pass  # raise not allowed exception

    def description(self) -> str:
        return self._datastore.description

    def set_description(self, description: str):
        self._datastore.description = description

    def get_image_uri(self, image_id: str) -> str:
        return self._dicomweb_client.get_object_url(self._datastore.objects[image_id])

    def get_label_uri(self, label_id: str) -> str:
        return self._dicomweb_client.get_object_url(self._datastore.objects[label_id])

    def get_image(self, image_id: str) -> Any:
        image = self._datastore.objects[image_id]
        nifti_output_path = os.path.join(self._datastore_path, f"{image_id}.nii.gz")

        _, nifti_file = ConverterUtil.to_nifti(
            self._dicomweb_client.get_object(image),
            nifti_output_path,
        )
        self._datastore.objects[image_id].local_path = nifti_file

        self._update_datastore_file()

        return io.BytesIO(pathlib.Path(nifti_output_path).read_bytes())

    def get_label(self, label_id: str) -> Any:
        return self.get_image(label_id)

    def get_label_by_image_id(self, image_id: str, tag: str) -> str:
        image = self._datastore.objects[image_id]
        for label_id in image.related_labels_keys:
            label = self._datastore.objects[label_id]
            if label.tag == tag:
                return label_id

        return ""

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        image = self._datastore.objects[image_id]
        return {label_id: self._datastore.objects[label_id].tag for label_id in image.related_labels_keys}

    def get_unlabeled_images(self) -> List[str]:
        image_ids = []
        images = {id: data for id, data in self._datastore.objects.items() if data.info['object_type'] == 'image'}
        for image_id, image_model in images.items():
            has_final_label = False
            for label_id in image_model.related_labels_keys:
                label = self._datastore.objects[label_id]
                if label.tag == DefaultLabelTag.FINAL.value:
                    has_final_label = True

            if not has_final_label:
                image_ids.append(image_id)

        return image_ids

    def add_image(self, image_id: str, image_filename: str) -> str:
        pass

    def datalist(self) -> List[Dict[str, str]]:
        pass

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        info = {}
        if self._datastore.objects[image_id].info:
            info.update(self._datastore.objects[image_id].info)

        # get the image from the DICOMWeb server so we can compute the checksum
        # if it's not been cached or is somehow not existent
        if not self._datastore.objects[image_id].local_path \
                or not os.path.exists(os.path.join(self._datastore_path, self._datastore.objects[image_id].local_path)):
            _ = self.get_image(image_id)

        local_path = os.path.join(self._datastore_path, self._datastore.objects[image_id].local_path)
        info.update({
            "checksum": file_checksum(pathlib.Path(local_path)),
            'name': self._datastore.objects[image_id].local_path,
            'path': local_path,
        })

        return info

    def get_label_info(self, label_id: str) -> Dict[str, Any]:
        pass

    def get_labeled_images(self) -> List[str]:
        pass

    def list_images(self) -> List[str]:
        pass

    def refresh(self) -> None:
        pass

    def remove_image(self, image_id: str) -> None:
        pass

    def remove_label(self, label_id: str) -> None:
        pass

    def remove_label_by_tag(self, label_tag: str) -> None:
        pass

    def save_label(self, image_id: str, label_filename: str, label_tag: str) -> str:
        pass

    def status(self) -> Dict[str, Any]:
        pass

    def update_image_info(self, image_id: str, info: Dict[str, Any]) -> None:
        pass

    def update_label_info(self, label_id: str, info: Dict[str, Any]) -> None:
        pass

    def __str__(self) -> str:
        return json.dumps(self._datastore.dict())

    def _update_datastore_file(self, lock=True):
        if lock:
            self._lock.acquire()

        logger.debug("+++ Updating datastore...")
        with open(self._datastore_config_path, "w") as f:
            f.write(json.dumps(self._datastore.dict(exclude_none=True, exclude_unset=True), indent=2, default=str))
        self._config_ts = os.stat(self._datastore_config_path).st_mtime

        if lock:
            self._lock.release()
