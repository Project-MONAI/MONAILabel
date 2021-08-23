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

import json
import logging
import os
import pathlib
import tempfile
import time
from typing import Any, Dict, List, Optional

import yaml
from dicomweb_client import DICOMwebClient
from lib import MyInfer, MyTrain
from lib.activelearning import MyStrategy, Tta
from monai.apps import load_from_mmar

from monailabel.interfaces import Datastore, MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.datastore.dicom.attributes import ATTRB_SOPINSTANCEUID
from monailabel.utils.datastore.dicom.util import binary_to_image, nifti_to_dicom_seg
from monailabel.utils.datastore.local import LocalDatastore
from monailabel.utils.others.generic import run_command
from monailabel.utils.scoring import Dice, Sum
from monailabel.utils.scoring.tta_scoring import TtaScoring

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.mmar = "clara_pt_spleen_ct_segmentation_1"

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Segmentation - Spleen",
            description="Active Learning solution to label Spleen Organ over 3D CT Images",
            version=2,
        )

    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_trainers(self):
        return {
            "segmentation_spleen": MyTrain(
                self.model_dir, load_from_mmar(self.mmar, self.model_dir), publish_path=self.final_model
            )
        }

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
            "Tta": Tta(),
        }

    def init_scoring_methods(self):
        return {
            "sum": Sum(),
            "dice": Dice(),
            "tta_scoring": TtaScoring(),
        }

    def init_datastore(self) -> Datastore:
        return MockStorage(self.studies, auto_reload=True)


# TODO:: This will be removed once DICOM Web support is added through datastore
class MockStorage(LocalDatastore):
    @staticmethod
    def from_dicom(dicom):
        logger.info(f"Temporary Hack:: Looking mapped image for: {dicom}")
        with open(os.path.join(os.path.dirname(__file__), "dicom.yaml"), "r") as fc:
            meta = yaml.full_load(fc)
            series_id = dicom["SeriesInstanceUID"]
            image_id = meta["series"][series_id]
            logger.info(f"Image Series: {series_id} => {image_id}")
            return image_id

    @staticmethod
    def to_dicom(image_id):
        logger.info(f"Temporary Hack:: Looking dicom info: {image_id}")
        with open(os.path.join(os.path.dirname(__file__), "dicom.yaml"), "r") as fc:
            meta = yaml.full_load(fc)
            return {
                "StudyInstanceUID": {v: k for k, v in meta.items()}[image_id],
                "SeriesInstanceUID": {v: k for k, v in meta.items()}[image_id],
            }

    def get_image_uri(self, image) -> str:
        image_id = self.from_dicom(json.loads(image))
        image_uri = super().get_image_uri(image_id)

        logger.info(f"Image ID : {image_id} => {image_uri}")
        return image_uri

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        res = super().get_image_info(image_id)
        res.update(self.to_dicom(image_id))

        logger.info(f"Image {image_id} => {res}")
        return res

    def save_label(self, image_id: str, label_filename: str, label_tag: str, label_info: Dict[str, Any]) -> str:
        logger.info(f"Input - Image Id: {image_id}")
        logger.info(f"Input - Label File: {label_filename}")
        logger.info(f"Input - Label Tag: {label_tag}")
        logger.info(f"Input - Label Info: {label_info}")

        image_uri = self.get_image_uri(image_id)
        logger.info(f"Image {image_uri}; Label: {label_filename}")

        label_ext = "".join(pathlib.Path(label_filename).suffixes)
        output_file = None
        if label_ext == ".bin":
            output_file = binary_to_image(image_uri, label_filename)
            label_filename = output_file

        logger.info(f"Label File: {output_file}")
        dicom = json.loads(image_id)
        res = super().save_label(self.from_dicom(dicom), label_filename, label_tag, label_info)

        # Create DICOM Seg and Upload to Orthanc
        with tempfile.TemporaryDirectory() as series_dir:
            download_series(dicom["StudyInstanceUID"], dicom["SeriesInstanceUID"], save_dir=series_dir)
            label_file = nifti_to_dicom_seg(series_dir, label_filename, label_info)

            run_command("curl", ["-X", "POST", "http://localhost:8042/instances", "--data-binary", f"@{label_file}"])
            os.unlink(label_file)

        if output_file:
            os.unlink(output_file)
        return res


class MyClient(DICOMwebClient):
    def _http_get_application_json(
        self, url: str, params: Optional[Dict[str, Any]] = None, stream: bool = False
    ) -> List[Dict[str, dict]]:
        content_type = "application/dicom+json"
        response = self._http_get(url, params=params, headers={"Accept": content_type}, stream=stream)
        if response.content:
            decoded_response: List[Dict[str, dict]] = response.json()
            if isinstance(decoded_response, dict):
                return [decoded_response]
            return decoded_response
        return []


def download_instance(study_id, series_id, instance_id, save_dir):
    file_name = os.path.join(save_dir, f"{instance_id}.dcm")
    client = MyClient(url="http://127.0.0.1:8042/dicom-web")
    instance = client.retrieve_instance(
        study_id,
        series_id,
        instance_id,
    )
    instance.save_as(file_name)


def download_series(study_id, series_id, save_dir):
    start = time.time()

    os.makedirs(save_dir, exist_ok=True)
    client = MyClient(url="http://127.0.0.1:8042/dicom-web")
    series_meta = client.retrieve_series_metadata(study_id, series_id)

    for idx, meta in enumerate(series_meta):
        instance_id = meta[ATTRB_SOPINSTANCEUID]["Value"][0]
        print(f"{idx} => {instance_id}")
        download_instance(study_id, series_id, instance_id, save_dir)

    logger.info(f"Time to download: {time.time() - start} (sec)")
