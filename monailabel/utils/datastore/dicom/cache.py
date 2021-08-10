import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from monailabel.interfaces import Datastore
from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.utils.datastore.dicom.client import DICOMWebClient
from monailabel.utils.datastore.dicom.convert import ConverterUtil
from monailabel.utils.datastore.dicom.datamodel import DICOMImageModel, DICOMLabelModel, DICOMWebDatastoreModel

logger = logging.getLogger(__name__)


class DicomWebCache(Datastore):
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
        self._dicomweb_client = dicomweb_client
        self._datastore_path = os.path.join(
            Path.home(),
            '.cache',
            'monailabel',
            hashlib.md5(self._dicomweb_uri.encode('utf-8')).hexdigest(),
        )
        self._datastore_config_path = os.path.join(self._datastore_path, datastore_config)
        self._label_path = label_store_path

        os.makedirs(self._datastore_path, exist_ok=True)
        os.makedirs(os.path.join(self._datastore_path, self._label_path), exist_ok=True)

        logger.info(f"DICOMWeb Endpoint: {self._dicomweb_uri}")
        logger.info(f"Datastore cache path: {self._datastore_path}")

        self._datastore: DICOMWebDatastoreModel = DICOMWebDatastoreModel(
            url=f"{self._dicomweb_uri}", description="Local Cache for DICOMWeb")

        self._datastore.objects = self._dicomweb_client.retrieve_dataset()

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

        nifti_vol, nifti_file = ConverterUtil.to_nifti(
            self._dicomweb_client.retrieve_series(image.study_id, image.series_id),
            nifti_output_path,
        )
        self._datastore.objects[image_id].local_path = nifti_file

        return nifti_vol

    def get_label(self, label_id: str) -> Any:
        return self.get_image(label_id)

    def get_label_by_image_id(self, image_id: str, tag: str) -> str:
        image = self._datastore.objects[image_id]
        image.__class__ = DICOMImageModel
        for label_id in image.related_labels:
            label = self._datastore.objects[label_id]
            if label.tag == tag:
                return label_id

        return ""

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        image = self._datastore.objects[image_id]
        image.__class__ = DICOMImageModel
        return {label_id: self._datastore.objects[label_id].tag for label_id in image.related_labels}

    def get_unlabeled_images(self) -> List[str]:
        image_ids = []
        for id, data in [obj for obj in self._datastore.objects if obj.info['object_type'] == 'image']:
            data.__class__ = DICOMImageModel
            has_final_label = False
            for label_id in data.related_labels:
                label = self._datastore[label_id]
                label.__class__ = DICOMLabelModel
                if label.tag == DefaultLabelTag.FINAL.value:
                    has_final_label = True

            if not has_final_label:
                image_ids.append(id)

        return image_ids

    def __str__(self) -> str:
        return json.dumps(self._datastore.dict())
