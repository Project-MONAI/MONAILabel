# Copyright (c) MONAI Consortium
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
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5

from dicomweb_client import DICOMwebClient
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread

from monailabel.utils.others.generic import run_command

logger = logging.getLogger(__name__)


def generate_key(patient_id: str, study_id: str, series_id: str):
    return md5(f"{patient_id}+{study_id}+{series_id}".encode()).hexdigest()


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


def dicom_web_download_series(study_id, series_id, save_dir, client: DICOMwebClient, frame_fetch=False):
    start = time.time()

    # Limitation for DICOMWeb Client as it needs StudyInstanceUID to fetch series
    if not study_id:
        meta = Dataset.from_json(
            [
                series
                for series in client.search_for_series(search_filters={"SeriesInstanceUID": series_id})
                if series["0020000E"]["Value"] == [series_id]
            ][0]
        )
        study_id = str(meta["StudyInstanceUID"].value)

    os.makedirs(save_dir, exist_ok=True)
    if not frame_fetch:
        instances = client.retrieve_series(study_id, series_id)
        for instance in instances:
            instance_id = str(instance["SOPInstanceUID"].value)
            file_name = os.path.join(save_dir, f"{instance_id}.dcm")
            instance.save_as(file_name)
    else:
        # TODO:: This logic (combining meta+pixeldata) needs improvement
        def save_from_frame(m):
            d = Dataset.from_json(m)
            instance_id = str(d["SOPInstanceUID"].value)

            # Hack to merge Info + RawData
            d.is_little_endian = True
            d.is_implicit_VR = True
            d.PixelData = client.retrieve_instance_frames(
                study_instance_uid=study_id,
                series_instance_uid=series_id,
                sop_instance_uid=instance_id,
                frame_numbers=[1],
            )[0]

            file_name = os.path.join(save_dir, f"{instance_id}.dcm")
            logger.info(f"++ Saved {os.path.basename(file_name)}")
            d.save_as(file_name)

        meta_list = client.retrieve_series_metadata(study_id, series_id)
        logger.info(f"++ Saving DCM into: {save_dir}")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="DICOMFetch") as executor:
            executor.map(save_from_frame, meta_list)

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


if __name__ == "__main__":
    import shutil

    from monailabel.datastore.dicom import DICOMwebClientX

    client = DICOMwebClientX(
        url="https://d1l7y4hjkxnyal.cloudfront.net",
        session=None,
        qido_url_prefix="output",
        wado_url_prefix="output",
        stow_url_prefix="output",
    )

    study_id = "1.2.840.113654.2.55.68425808326883186792123057288612355322"
    series_id = "1.2.840.113654.2.55.257926562693607663865369179341285235858"

    save_dir = "/local/sachi/Data/dicom"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    dicom_web_download_series(study_id, series_id, save_dir, client, True)
