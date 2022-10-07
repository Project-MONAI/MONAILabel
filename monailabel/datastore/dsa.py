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
import logging
import os
import pathlib
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import girder_client
import numpy as np
from PIL import Image

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag

logger = logging.getLogger(__name__)


class DSADatastore(Datastore):
    def __init__(self, api_url, api_key=None, folder=None, annotation_groups=None, asset_store_path="", cache_path=""):
        self.api_url = api_url
        self.api_key = api_key
        self.folders = folder.split(",") if folder else []
        self.folders = {f.strip() for f in self.folders}
        self.annotation_groups = [a.lower() if a else a for a in annotation_groups] if annotation_groups else []
        self.asset_store_path = asset_store_path

        uri_hash = hashlib.md5(api_url.encode("utf-8")).hexdigest()
        self.cache_path = (
            os.path.join(cache_path, uri_hash)
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "dsa", uri_hash)
        )

        logger.info(f"DSA:: Api Url: {api_url}")
        logger.info(f"DSA:: Api Key: {'*' * len(api_key) if api_key else ''}")
        logger.info(f"DSA:: Folder (Images): {folder}")
        logger.info(f"DSA:: Annotation Groups: {annotation_groups}")
        logger.info(f"DSA:: Local Asset Store Path: {asset_store_path}")

        self.gc = girder_client.GirderClient(apiUrl=api_url)
        if api_key:
            self.gc.authenticate(apiKey=api_key)

    def name(self) -> str:
        return "DSA Datastore"

    def set_name(self, name: str):
        pass

    def description(self) -> str:
        return "Digital Slide Archive"

    def set_description(self, description: str):
        pass

    def datalist(self) -> List[Dict[str, Any]]:
        return [
            {
                "api_url": self.api_url,
                "image": image_id,
                "label": image_id,
                "groups": self.annotation_groups,
            }
            for image_id in self.get_labeled_images()
        ]

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        return {DefaultLabelTag.FINAL.name: image_id}

    def get_label_by_image_id(self, image_id: str, tag: str) -> str:
        return image_id

    def get_image(self, image_id: str, params=None) -> Any:
        try:
            name = self.get_image_info(image_id)["name"]
        except girder_client.HttpError:
            image_id, name = self._name_to_id(image_id)

        location = params.get("location", [0, 0])
        size = params.get("size", [0, 0])
        if sum(location) <= 0 and sum(size) <= 0:  # whole side image
            dest = os.path.join(self.cache_path, name)
            if not os.path.exists(dest):
                logger.info(f"Downloading: {image_id} => {name} => {dest}")
                self.gc.downloadItem(itemId=image_id, dest=self.cache_path)
            return dest

        parameters = {
            "left": location[0],
            "top": location[1],
            "regionWidth": size[0],
            "regionHeight": size[1],
            "units": "base_pixels",
            "encoding": "PNG",
        }

        resp = self.gc.get(f"item/{image_id}/tiles/region", parameters=parameters, jsonResp=False)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _name_to_id(self, name):
        folders = self.folders if self.folders else self._get_all_folders()
        for folder in folders:
            data = self.gc.get("item", parameters={"folderId": folder, "limit": 0})
            for d in data:
                if d.get("largeImage") and d["name"] == name or Path(d["name"]).stem == name:
                    return d["_id"], d["name"]
        return name

    def get_image_uri(self, image_id: str) -> str:
        try:
            name = self.get_image_info(image_id)["name"]
        except girder_client.HttpError:
            image_id, name = self._name_to_id(image_id)

        if self.asset_store_path:
            data = self.gc.get(f"item/{image_id}/files", parameters={"limit": 0})
            assets = [d["assetstoreId"] for d in data]
            for asset in assets:
                files = self.gc.get(f"assetstore/{asset}/files", parameters={"limit": 0})
                for f in files:
                    if f["itemId"] == image_id:
                        return str(os.path.join(self.asset_store_path, f["path"]))
        else:
            cached = os.path.join(self.cache_path, name)
            if os.path.exists(cached):
                return str(cached)

        return f"{self.api_url}/item/{image_id}"

    def get_label(self, label_id: str, label_tag: str, params=None) -> Any:
        return self.gc.get(f"annotation/item/{label_id}")

    def get_label_uri(self, label_id: str, label_tag: str) -> str:
        return f"{self.api_url}/annotation/item/{label_id}"

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        return self.gc.getItem(image_id)  # type: ignore

    def get_label_info(self, label_id: str, label_tag: str) -> Dict[str, Any]:
        return {}

    def _get_annotated_images(self):
        data = self.gc.get("annotation", parameters={"limit": 0})

        images = []
        for d in data:
            if not self.annotation_groups:
                images.append(d["itemId"])
                continue

            # get annotations and find if any matching groups exist
            matched = [
                g for g in d["groups"] if g in self.annotation_groups or (g and g.lower() in self.annotation_groups)
            ]
            if matched:
                images.append(d["itemId"])
        return images

    def get_labeled_images(self) -> List[str]:
        images = self.list_images()
        annotated = self._get_annotated_images()
        return [image for image in images if image in annotated]

    def get_unlabeled_images(self) -> List[str]:
        images = self.list_images()
        labeled = self.get_labeled_images()
        return [image for image in images if image not in labeled]

    def _get_all_folders(self):
        folders = []
        for collection in self.gc.listCollection():
            for folder in self.gc.listFolder(parentId=collection["_id"], parentFolderType="collection"):
                folders.append(folder["_id"])
        return folders

    def list_images(self) -> List[str]:
        images = []
        folders = self.folders if self.folders else self._get_all_folders()
        for folder in folders:
            for item in self.gc.get("item", parameters={"folderId": folder, "limit": 0}):
                if item.get("largeImage"):
                    images.append(item["_id"])
        return images

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

    def get_dataset_archive(self, limit_cases: Optional[int]) -> str:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {
            "total": len(self.list_images()),
            "completed": len(self.get_labeled_images()),
        }

    def json(self):
        return self.datalist()


def main():
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    api_url = "http://0.0.0.0:8080/api/v1"
    folder = None
    annotation_groups = None
    asset_store_path = None
    api_key = None  # "zBsr184BByiRK0BUyMMB01v3O8kTqkXPbqxndpfi"

    # api_url = "https://demo.kitware.com/histomicstk/api/v1"
    # folder = "5bbdeba3e629140048d017bb"
    # annotation_groups = ["mostly_tumor"]
    # asset_store_path = None
    # api_key = None

    ds = DSADatastore(
        api_url=api_url,
        api_key=api_key,
        folder=folder,
        annotation_groups=annotation_groups,
        asset_store_path=asset_store_path,
    )

    images = ds.list_images()
    print(f"Images: {images}")

    labeled_images = ds.get_labeled_images()
    print(f"Labeled Images: {labeled_images}")

    unlabeled_images = ds.get_unlabeled_images()
    print(f"UnLabeled Images: {unlabeled_images}")

    image_id = images[0]
    print(f"Image Info: {json.dumps(ds.get_image_info(image_id), indent=2)}")
    print(f"Image URI: {ds.get_image_uri(image_id)}")
    print(f"Image URI (name): {ds.get_image_uri('TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7')}")
    print(f"Labels: {ds.get_labels_by_image_id(image_id)}")

    if labeled_images:
        label_id = labeled_images[0]
        label_tag = "FINAL"
        print(f"Label Info: {json.dumps(ds.get_label_info(label_id, label_tag), indent=2)}")
        print(f"Label URI: {ds.get_label_uri(label_id, label_tag)}")

    print(f"Dataset for Training: \n{json.dumps(ds.datalist(), indent=2)}")

    img = ds.get_image(
        "TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7",
        params={
            "location": (6090, 15863),
            "size": (1071, 714),
        },
    )
    print(f"Fetched Region: {img.shape}")


if __name__ == "__main__":
    main()
