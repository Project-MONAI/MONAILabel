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

import hashlib
import io
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests
from requests.auth import HTTPBasicAuth

from monailabel.interfaces.datastore import Datastore

logger = logging.getLogger(__name__)
xnat_ns = {"xnat": "http://nrg.wustl.edu/xnat"}


class XNATDatastore(Datastore):
    def __init__(self, api_url, username=None, password=None, project=None, asset_path="", cache_path=""):
        self.api_url = api_url
        self.xnat_session = requests.sessions.session()
        self.auth = HTTPBasicAuth(username, password) if username else None
        self.xnat_csrf = ""
        self._login_xnat()

        self.projects = project.split(",") if project else []
        self.projects = {p.strip() for p in self.projects}
        self.asset_path = asset_path

        uri_hash = hashlib.md5(api_url.encode("utf-8")).hexdigest()
        cache_path = cache_path.strip() if cache_path else ""
        self.cache_path = (
            os.path.join(cache_path, uri_hash)
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "xnat", uri_hash)
        )

        logger.info(f"XNAT:: API URL: {api_url}")
        logger.info(f"XNAT:: UserName: {username}")
        logger.info(f"XNAT:: Password: {'*' * len(password) if password else ''}")
        logger.info(f"XNAT:: Project: {project}")
        logger.info(f"XNAT:: AssetPath: {asset_path}")

    def name(self) -> str:
        return "XNAT Datastore"

    def set_name(self, name: str):
        pass

    def description(self) -> str:
        return "XNAT Datastore"

    def set_description(self, description: str):
        pass

    def datalist(self) -> List[Dict[str, Any]]:
        return [
            {
                "api_url": self.api_url,
                "image": image_id,
                "label": image_id,
            }
            for image_id in self.get_labeled_images()
        ]

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        pass

    def get_label_by_image_id(self, image_id: str, tag: str) -> str:
        pass

    def get_image(self, image_id: str, params=None) -> Any:
        p = self._download_image(image_id, check_zip=True)
        uri = os.path.join(os.path.dirname(p), "files.zip")
        return io.BytesIO(pathlib.Path(uri).read_bytes()) if uri else None

    def get_image_uri(self, image_id: str) -> str:
        return self._download_image(image_id, check_zip=False)

    def get_label(self, label_id: str, label_tag: str, params=None) -> Any:
        pass

    def get_label_uri(self, label_id: str, label_tag: str) -> str:
        pass

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        info = {}

        project, subject, experiment, scan = self._id_to_fields(image_id)
        url = "{}/data/projects/{}/subjects/{}/experiments/{}/scans/{}?format=xml".format(
            self.api_url,
            quote_plus(project),
            quote_plus(subject),
            quote_plus(experiment),
            quote_plus(scan),
        )

        response = self._request_get(url)
        if response.ok:
            info.update({"project": project, "subject": subject, "experiment": experiment, "scan": scan})
        return info

    def get_label_info(self, label_id: str, label_tag: str) -> Dict[str, Any]:
        return {}

    def get_labeled_images(self) -> List[str]:
        return []

    def get_unlabeled_images(self) -> List[str]:
        return self.list_images()

    def list_images(self) -> List[str]:
        image_ids: List[str] = []

        response = self._request_get(f"{self.api_url}/data/projects?format=json")
        for p in response.json().get("ResultSet", {}).get("Result", []):
            project = p.get("ID")
            if self.projects and project not in self.projects:
                continue

            response = self._request_get(f"{self.api_url}/data/projects/{quote_plus(project)}/experiments?format=json")
            for e in response.json().get("ResultSet", {}).get("Result", []):
                experiment = e.get("ID")

                response = self._request_get(f"{self.api_url}/data/experiments/{quote_plus(experiment)}?format=xml")
                tree = ElementTree.fromstring(response.content)
                s = tree.find(".//xnat:subject_ID", namespaces=xnat_ns)
                if s is None:
                    continue

                subject = s.text
                for n in tree.findall(".//xnat:scan", namespaces=xnat_ns):
                    scan = n.get("ID")
                    image_ids.append(f"{project}/{subject}/{experiment}/{scan}")

        return image_ids

    def refresh(self) -> None:
        pass

    def add_image(self, image_id: str, image_filename: str, image_info: Dict[str, Any]) -> str:
        raise NotImplementedError

    def remove_image(self, image_id: str) -> None:
        raise NotImplementedError

    def save_label(self, image_id: str, label_filename: str, label_tag: str, label_info: Dict[str, Any]) -> str:
        raise NotImplementedError

    def remove_label(self, label_id: str, label_tag: str) -> None:
        raise NotImplementedError

    def update_image_info(self, image_id: str, info: Dict[str, Any]) -> None:
        pass

    def update_label_info(self, label_id: str, label_tag: str, info: Dict[str, Any]) -> None:
        pass

    def get_dataset_archive(self, limit_cases: Optional[int]) -> str:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {
            "total": len(self.list_images()),
            "completed": len(self.get_labeled_images()),
        }

    def json(self):
        return self.datalist()

    def _find_in_asset_store(self, project, subject, experiment, scan) -> str:
        url = "{}/data/projects/{}/subjects/{}/experiments/{}/scans/{}?format=xml".format(
            self.api_url,
            quote_plus(project),
            quote_plus(subject),
            quote_plus(experiment),
            quote_plus(scan),
        )

        response = self._request_get(url)
        if response.ok:
            tree = ElementTree.fromstring(response.content)
            ele = tree.find('.//xnat:file[@label="DICOM"]', namespaces=xnat_ns)
            path = ele.get("URI") if ele is not None else ""
            if path:
                dicom_dir = os.path.dirname(os.path.join(self.asset_path, path.replace("/data/xnat/archive/", "")))
                if os.path.exists(dicom_dir) and len(os.listdir(dicom_dir)) > 0:
                    return dicom_dir

        return ""

    def _download_zip(self, dest_dir, dest_zip, dicom_dir, project, subject, experiment, scan):
        url = "{}/data/projects/{}/subjects/{}/experiments/{}/scans/{}/files?format=zip".format(
            self.api_url,
            quote_plus(project),
            quote_plus(subject),
            quote_plus(experiment),
            quote_plus(scan),
        )

        response = self._request_get(url)
        if not response.ok:
            logger.info(f"Image Fetch Failed: {response.status_code} {response.reason}")
            return ""

        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_zip, "wb") as fp:
            fp.write(response.content)

        extract_dir = os.path.join(dest_dir, "temp")
        shutil.unpack_archive(dest_zip, extract_dir)

        os.makedirs(dicom_dir, exist_ok=True)
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".dcm"):
                    shutil.move(os.path.join(root, f), dicom_dir)

        shutil.rmtree(extract_dir)
        return dicom_dir

    def _download_image(self, image_id, check_zip=False) -> str:
        project, subject, experiment, scan = self._id_to_fields(image_id)
        if self.projects and project not in self.projects:
            logger.info(f"Access to Project: {project} is restricted;  Allowed: {self.projects}")
            return ""

        # Check in Asset Store
        if self.asset_path and not check_zip:
            dicom_dir = self._find_in_asset_store(project, subject, experiment, scan)
            if dicom_dir:
                logger.info(f"Exists in asset store: {self.asset_path}")
                return dicom_dir

        # Check in Local Cache
        dest_dir = os.path.join(self.cache_path, project, subject, experiment, scan)
        dest_zip = os.path.join(dest_dir, "files.zip")
        dicom_dir = os.path.join(dest_dir, "DICOM")
        if os.path.exists(dest_zip) and len(os.listdir(dicom_dir)) > 0:
            logger.info(f"Exists in cache: {dest_zip}")
            return dicom_dir

        # Download DICOM Zip
        logger.info(f"Downloading: {project} => {subject} => {experiment} => {scan} => {dest_zip}")
        start = time.time()

        self._download_zip(dest_dir, dest_zip, dicom_dir, project, subject, experiment, scan)
        logger.info(f"Download Time (ms) for {image_id}: {round(time.time() - start, 4)}")
        return dicom_dir

    def _id_to_fields(self, image_id):
        fields = image_id.split("/")
        project = fields[0]
        subject = fields[1]
        experiment = fields[2]
        scan = fields[3]
        return project, subject, experiment, scan

    def _login_xnat(self):
        # Get CSRF token
        url = "{}/data/JSESSION?CSRF=true".format(
            self.api_url,
        )
        csrf_response = self._request_get(url)
        if not csrf_response.ok:
            logger.error("XNAT:: Could not get XNAT CSRF token")
            raise Exception("Could not get XNAT CSRF token")
        content = csrf_response.content
        self.xnat_csrf = content.decode("utf-8").strip().split("=")[1]

        # Log in to XNAT
        url = f"{self.api_url}/data/JSESSION?XNAT_CSRF={self.xnat_csrf}"
        login_response = self.xnat_session.post(url, auth=self.auth, allow_redirects=True)
        if not login_response.ok:
            logger.error("XNAT:: Could not log in to XNAT")
            raise Exception("Could not log in to XNAT")

        logger.info("XNAT:: Logged in XNAT")

    def _request_get(self, url):
        return self.xnat_session.get(url, allow_redirects=True)


def main():
    from monai.transforms import LoadImage

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create alias token for user through instead of using direct username and password
    # http://127.0.0.1/data/services/tokens/issue
    ds = XNATDatastore(
        api_url="http://127.0.0.1",
        username="admin",  # "a8a73c8d-a0bd-44d1-87af-244476072af4",
        password="admin",  # "wGdawXhqo9Fhsh5p1pd6nGRloF99mxXYvvBGjCtTl1A9zYkk4mlaQJuvJQUcXL62",
        asset_path="/localhome/sachi/Projects/xnat-docker-compose/xnat-data/archive",
        project="Test",
    )

    image_ids = ds.list_images()
    logger.info("\n" + "\n".join(image_ids))

    image_id = "Test/XNAT01_S00003/XNAT01_E00004/1_2_826_0_1_3680043_8_274_1_1_8323329_10631_1656479315_17615"
    image_uri = ds.get_image_uri(image_id)
    logger.info(f"+++ Image URI: {image_uri}")

    if image_uri:
        loader = LoadImage(image_only=True)
        image_np = loader(image_uri)
        logger.info(f"Image Shape: {image_np.shape}")


if __name__ == "__main__":
    main()
