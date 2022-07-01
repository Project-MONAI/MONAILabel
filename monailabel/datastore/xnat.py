import hashlib
import io
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List
from urllib.parse import quote_plus

import requests

from monailabel.interfaces.datastore import Datastore

logger = logging.getLogger(__name__)


class XNATDatastore(Datastore):
    def __init__(self, url, username, password, project, asset_path="", cache_path=""):
        self.url = url
        self.username = username
        self.password = password
        self.project = project
        self.asset_path = asset_path

        uri_hash = hashlib.md5(url.encode("utf-8")).hexdigest()

        cache_path = cache_path.strip() if cache_path else ""
        self.cache_path = (
            os.path.join(cache_path, uri_hash)
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "xnat", uri_hash)
        )

        logger.info(f"XNAT:: URL: {url}")
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
        return []

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        pass

    def get_label_by_image_id(self, image_id: str, tag: str) -> str:
        pass

    def get_image(self, image_id: str, params=None) -> Any:
        uri = os.path.join(os.path.dirname(self.get_image_uri(image_id)), "files.zip")
        return io.BytesIO(pathlib.Path(uri).read_bytes()) if uri else None

    def get_image_uri(self, image_id: str) -> str:
        fields = image_id.split("/")
        project = fields[0]
        subject = fields[1]
        experiment = fields[2]
        scan = fields[3]

        dest_dir = os.path.join(self.cache_path, project, subject, experiment, scan)
        dest_zip = os.path.join(dest_dir, "files.zip")
        dicom_dir = os.path.join(dest_dir, "dicom")
        if os.path.exists(dest_zip) and len(os.listdir(dicom_dir)) > 0:
            logger.info(f"Exists in cache: {dest_zip}")
            return dicom_dir

        logger.info(f"Downloading: {project} => {subject} => {experiment} => {scan} => {dest_zip}")
        session = requests.Session()
        if self.username:
            session.auth = (self.username, self.password)

        url = "{}/data/projects/{}/subjects/{}/experiments/{}/scans/{}/files?format=zip".format(
            self.url,
            quote_plus(project),
            quote_plus(subject),
            quote_plus(experiment),
            quote_plus(scan),
        )
        response = session.get(url, allow_redirects=True)

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

    def get_label(self, label_id: str, label_tag: str, params=None) -> Any:
        pass

    def get_label_uri(self, label_id: str, label_tag: str) -> str:
        pass

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        pass

    def get_label_info(self, label_id: str, label_tag: str) -> Dict[str, Any]:
        pass

    def get_labeled_images(self) -> List[str]:
        return []

    def get_unlabeled_images(self) -> List[str]:
        return []

    def list_images(self) -> List[str]:
        return []

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
        raise NotImplementedError

    def update_label_info(self, label_id: str, label_tag: str, info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {
            "total": len(self.list_images()),
            "completed": len(self.get_labeled_images()),
        }

    def json(self):
        return self.datalist()


def main():
    from monai.transforms import LoadImage

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    url = "http://127.0.0.1"
    username = "admin"
    password = "admin"
    project = "Test"
    asset_store_path = "/localhome/sachi/Projects/xnat-docker-compose/"
    ds = XNATDatastore(url, username, password, project, asset_store_path)

    image_id = "Test/XNAT01_S00003/XNAT01_E00004/1_2_826_0_1_3680043_8_274_1_1_8323329_10631_1656479315_17615"
    image_uri = ds.get_image_uri(image_id)
    logger.info(f"+++ Image URI: {image_uri}")

    loader = LoadImage(image_only=True)
    image_np = loader(image_uri)
    logger.info(f"Image Shape: {image_np.shape}")


if __name__ == "__main__":
    main()
