# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import io
import json
import logging
import os
import pathlib
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import nibabel
import pydicom
from filelock import FileLock

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.utils.datastore.dicom.attributes import (
    ATTRB_MONAILABELINDICATOR,
    ATTRB_MONAILABELTAG,
    ATTRB_SERIESINSTANCEUID,
    str2hex,
)
from monailabel.utils.datastore.dicom.client import DICOMWebClient
from monailabel.utils.datastore.dicom.convert import ConverterUtil
from monailabel.utils.datastore.dicom.datamodel import DICOMObjectModel, DICOMWebDatastoreModel
from monailabel.utils.datastore.dicom.util import generate_key
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
            ".cache",
            "monailabel",
            hashlib.md5(dicomweb_uri.encode("utf-8")).hexdigest(),
        )
        self._lock = FileLock(os.path.join(self._datastore_path, ".lock"))
        self._datastore_config_path = os.path.join(self._datastore_path, datastore_config)
        self._label_store_path = label_store_path

        os.makedirs(self._datastore_path, exist_ok=True)
        os.makedirs(os.path.join(self._datastore_path, self._label_store_path), exist_ok=True)

        logger.info(f"DICOMWeb Endpoint: {dicomweb_uri}")
        logger.info(f"Datastore cache path: {self._datastore_path}")

        self._datastore: DICOMWebDatastoreModel = DICOMWebDatastoreModel(
            url=f"{dicomweb_uri}", description="Local Cache for DICOMWeb"
        )

        self._datastore.objects = self._dicomweb_client.retrieve_dataset()
        local_cache = self._load_existing_datastore_config()
        self._reconcile_local_datastore(local_cache)
        self._update_datastore_file()

    def name(self) -> str:
        return self._datastore.url

    def set_name(self, name: str):
        pass  # TODO: raise not allowed exception

    def description(self) -> str:
        return self._datastore.description

    def set_description(self, description: str):
        self._datastore.description = description

    def get_image_uri(self, image_id: str) -> str:
        # get the image from the DICOMWeb server so we can compute the checksum
        # if it's not been cached or is somehow not existent
        if not self._datastore.objects[image_id].local_path or not os.path.exists(
            os.path.join(self._datastore_path, self._datastore.objects[image_id].local_path)
        ):
            _ = self.get_image(image_id)

        return os.path.join(self._datastore_path, self._datastore.objects[image_id].local_path)

    def get_label_uri(self, label_id: str) -> str:
        return self.get_image_uri(label_id)

    def get_image(self, image_id: str) -> Any:
        image = self._datastore.objects[image_id]
        nifti_output_path = os.path.join(self._datastore_path, f"{image_id}.nii.gz")
        if not os.path.exists(nifti_output_path) or not self._datastore.objects[image_id].memory_cache.get(
            "dicom_dataset"
        ):
            instances = self._dicomweb_client.get_object(image)
            self._datastore.objects[image_id].memory_cache.update(
                {
                    "dicom_dataset": instances,
                }
            )

            with FileLock(f"{nifti_output_path}.lock"):
                _, nifti_file = ConverterUtil.to_nifti(instances, nifti_output_path)

            self._datastore.objects[image_id].local_path = nifti_file
        else:
            self._datastore.objects[image_id].local_path = nifti_output_path

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
        images = {id: data for id, data in self._datastore.objects.items() if data.info["object_type"] == "image"}
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

    def datalist(self, full_path=True) -> List[Dict[str, str]]:
        items = []
        images = {
            image_id: image
            for image_id, image in self._datastore.objects.items()
            if image.info["object_type"] == "image"
        }
        for image_id, image in images.items():

            image_path = self._get_path(image.local_path, False, full_path) if image.local_path else image_id

            for label_key in image.related_labels_keys:
                label = self._datastore.objects[label_key]
                if label.tag == DefaultLabelTag.FINAL:
                    items.append(
                        {
                            "image": image_path,
                            "label": self._get_path(label.local_path, True, full_path)
                            if label.local_path
                            else label_key,
                        }
                    )
        return items

    def _get_path(self, path: str, is_label: bool, full_path=True):
        if is_label:
            path = os.path.join(self._label_store_path, path)

        if not full_path or os.path.isabs(path):
            return path

        return os.path.realpath(os.path.join(self._datastore_path, path))

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        info = {}
        if self._datastore.objects[image_id].info:
            info.update(self._datastore.objects[image_id].info)

        # get the image from the DICOMWeb server so we can compute the checksum
        # if it's not been cached or is somehow not existent
        if not self._datastore.objects[image_id].local_path or not os.path.exists(
            os.path.join(self._datastore_path, self._datastore.objects[image_id].local_path)
        ):
            _ = self.get_image(image_id)

        local_path = os.path.join(self._datastore_path, self._datastore.objects[image_id].local_path)
        info.update(
            {
                "patient_id": self._datastore.objects[image_id].patient_id,
                "study_id": self._datastore.objects[image_id].study_id,
                "series_id": self._datastore.objects[image_id].series_id,
                "checksum": file_checksum(pathlib.Path(local_path)),
                "name": self._datastore.objects[image_id].local_path,
                "path": local_path,
            }
        )

        return info

    def get_label_info(self, label_id: str) -> Dict[str, Any]:
        return self.get_image_info(label_id)

    def get_labeled_images(self) -> List[str]:
        return [
            object_id
            for object_id, obj in self._datastore.objects.items()
            if obj.info["object_type"] == "image" and not obj.related_labels_keys
        ]

    def list_images(self) -> List[str]:
        return [object_id for object_id, obj in self._datastore.objects.items() if obj.info["object_type"] == "image"]

    def refresh(self) -> None:
        pass

    def remove_image(self, image_id: str) -> None:
        pass

    def remove_label(self, label_id: str) -> None:
        pass

    def remove_label_by_tag(self, label_tag: str) -> None:
        pass

    def save_label(self, image_id: str, label_filename: str, label_tag: str, label_info: Dict[str, Any]) -> str:
        logger.info(f"Saving Label for Image: {image_id}; Tag: {label_tag}")

        image = self._datastore.objects[image_id]

        # get the dicom image dataset to use as the template for generating DICOMSEG
        # from inference result
        original_dataset = None
        if not image.memory_cache.get("dicom_dataset"):
            _ = self.get_image(image_id)
        original_dataset = image.memory_cache["dicom_dataset"]

        # convert segmentation result in `label_filename` to a numpy array
        seg_image = nibabel.load(label_filename)

        label_names: List[str] = []
        if label_info and label_info.get("label_names"):
            label_names = [item["name"] for item in label_info["label_names"]]

        dcmseg_dataset = ConverterUtil.to_dicom(original_dataset, seg_image.get_fdata(), label_names)
        series_id = dcmseg_dataset[str2hex(ATTRB_SERIESINSTANCEUID)].value
        label_id: str = generate_key(image.patient_id, image.study_id, series_id)

        # add label tag to DICOMSEG image in MONAI Label private tag `ATTRB_MONAILABELTAG`
        dcmseg_dataset.add_new(str2hex(ATTRB_MONAILABELTAG), "LO", label_tag)
        dcmseg_dataset.add_new(str2hex(ATTRB_MONAILABELINDICATOR), "CS", "Y")

        # send the new DICOMSEG label to the DICOMWeb server
        if isinstance(dcmseg_dataset, pydicom.Dataset):
            dcmseg_dataset = [dcmseg_dataset]

        self._dicomweb_client.push_series(image, dcmseg_dataset)

        datastore_label_path = os.path.join(
            self._datastore_path, self._label_store_path, f"{label_id}{''.join(pathlib.Path(label_filename).suffixes)}"
        )
        shutil.copy(src=label_filename, dst=datastore_label_path, follow_symlinks=True)
        label = DICOMObjectModel(
            patient_id=image.patient_id,
            study_id=image.study_id,
            series_id=series_id,
            tag=label_tag,
            local_path=datastore_label_path,
            info={
                "object_type": "label",
                "ts": int(time.time()),
            },
        )

        # add the newly created label reference to the image from which it was generated
        self._datastore.objects[image_id].related_labels_keys.append(label_id)

        self._datastore.objects.update(
            {
                label_id: label,
            }
        )

        self._update_datastore_file()

        return label_id

    def status(self) -> Dict[str, Any]:
        labels: List[DICOMObjectModel] = []
        tags: dict = {}
        for label in labels:
            tags[label.tag] = tags.get(label.tag, 0) + 1

        return {
            "total": len(self.list_images()),
            "completed": len(self.get_labeled_images()),
            "label_tags": tags,
            "train": self.datalist(full_path=False),
        }

    def update_image_info(self, image_id: str, info: Dict[str, Any]) -> None:
        pass

    def update_label_info(self, label_id: str, info: Dict[str, Any]) -> None:
        pass

    def __str__(self) -> str:
        return json.dumps(self._datastore.dict())

    def _load_existing_datastore_config(self):
        existing_datastore = None
        with self._lock:
            if os.path.exists(self._datastore_config_path):
                existing_datastore = DICOMWebDatastoreModel.parse_file(self._datastore_config_path)

        return existing_datastore

    def _reconcile_local_datastore(self, local_datastore: DICOMWebDatastoreModel):

        if not local_datastore:
            return

        # find all the keys that already exist in the locally cached datastore
        # and put them into the datastore that was newly fathed from the dicomweb connection
        for object_id in self._datastore.objects.keys():
            if object_id in local_datastore.objects.keys():
                self._datastore.objects[object_id] = local_datastore.objects[object_id]

        for obj in self._datastore.objects.values():
            if obj.info.get("object_type") == "image":
                exclude_label_keys = []
                for label_key in obj.related_labels_keys:
                    if label_key not in self._datastore.objects.keys():
                        exclude_label_keys.append(label_key)
                obj.related_labels_keys = [k for k in obj.related_labels_keys if k not in exclude_label_keys]

    def _update_datastore_file(self, lock=True):
        if lock:
            self._lock.acquire()

        logger.debug("+++ Updating datastore...")
        with open(self._datastore_config_path, "w") as f:
            f.write(
                json.dumps(
                    self._datastore.dict(
                        exclude_none=True,
                        exclude_unset=True,
                        exclude={"memory_cache"},
                    ),
                    indent=2,
                    default=str,
                )
            )
        self._config_ts = os.stat(self._datastore_config_path).st_mtime

        if lock:
            self._lock.release()

    def json(self):
        return self._datastore.dict()
