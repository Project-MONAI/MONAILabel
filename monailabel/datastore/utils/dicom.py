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
import logging
import os
import time
from hashlib import md5

from dicomweb_client import DICOMwebClient
from dicomweb_client.api import load_json_dataset
from pydicom.filereader import dcmread

from monailabel.utils.others.generic import run_command

logger = logging.getLogger(__name__)


def generate_key(patient_id: str, study_id: str, series_id: str):
    return md5(f"{patient_id}+{study_id}+{series_id}".encode("utf-8")).hexdigest()


def get_scu(query, output_dir, query_level="SERIES", host="127.0.0.1", port="4242", aet="MONAILABEL"):
    start = time.time()
    field = "StudyInstanceUID" if query_level == "STUDIES" else "SeriesInstanceUID"
    run_command(
        "python",
        [
            "-m",
            "pynetdicom",
            "getscu",
            host,
            port,
            "-P",
            "-k",
            f"0008,0052={query_level}",
            "-k",
            f"{field}={query}",
            "-aet",
            aet,
            "-q",
            "-od",
            output_dir,
        ],
    )
    logger.info(f"Time to run GET-SCU: {time.time() - start} (sec)")


def store_scu(input_file, host="127.0.0.1", port="4242", aet="MONAILABEL"):
    start = time.time()
    input_files = input_file if isinstance(input_file, list) else [input_file]
    for i in input_files:
        run_command("python", ["-m", "pynetdicom", "storescu", host, port, "-aet", aet, i])
    logger.info(f"Time to run STORE-SCU: {time.time() - start} (sec)")


def dicom_web_download_series(study_id, series_id, save_dir, client: DICOMwebClient):
    start = time.time()

    # Limitation for DICOMWeb Client as it needs StudyInstanceUID to fetch series
    if not study_id:
        meta = load_json_dataset(client.search_for_series(search_filters={"SeriesInstanceUID": series_id})[0])
        study_id = str(meta["StudyInstanceUID"].value)

    os.makedirs(save_dir, exist_ok=True)
    instances = client.retrieve_series(study_id, series_id)
    for instance in instances:
        instance_id = str(instance["SOPInstanceUID"].value)
        file_name = os.path.join(save_dir, f"{instance_id}.dcm")
        instance.save_as(file_name)

    logger.info(f"Time to download: {time.time() - start} (sec)")


def dicom_web_upload_dcm(input_file, client: DICOMwebClient):
    start = time.time()
    dataset = dcmread(input_file)
    result = client.store_instances([dataset])

    url = ""
    for elm in result.iterall():
        s = str(elm.value)
        logger.info(f"{s}")
        if "/series/" in s:
            url = s
            break

    series_id = url.split("/series/")[1].split("/")[0] if url else ""
    logger.info(f"Series Instance UID: {series_id}")

    logger.info(f"Time to upload: {time.time() - start} (sec)")
    return series_id
