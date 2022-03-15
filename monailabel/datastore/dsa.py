import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import girder_client

from monailabel.interfaces.datastore import Datastore, DefaultLabelTag

logger = logging.getLogger(__name__)


class DSADatastore(Datastore):
    def __init__(self, api_url, folder, api_key=None, annotation_groups=None, asset_store_path=""):
        self.api_url = api_url
        self.folder = folder
        self.annotation_groups = [a.lower() if a else a for a in annotation_groups] if annotation_groups else []
        self.asset_store_path = asset_store_path

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

    def get_image(self, image_id: str) -> Any:
        return self.gc.get(f"/item/{image_id}")

    def name_to_id(self, name):
        data = self.gc.get("item", parameters={"folderId": self.folder, "limit": 0})
        for d in data:
            if d.get("largeImage") and d["name"] == name or Path(d["name"]).stem == name:
                return d["_id"]
        return name

    def get_image_uri(self, image_id: str) -> str:
        try:
            self.get_image_info(image_id)
        except girder_client.HttpError:
            image_id = self.name_to_id(image_id)

        if self.asset_store_path:
            data = self.gc.get(f"/item/{image_id}/files", parameters={"limit": 0})
            assets = [d["assetstoreId"] for d in data]
            for asset in assets:
                files = self.gc.get(f"/assetstore/{asset}/files", parameters={"limit": 0})
                for f in files:
                    if f["itemId"] == image_id:
                        return str(os.path.join(self.asset_store_path, f["path"]))

        return f"{self.api_url}/item/{image_id}"

    def get_label(self, label_id: str, label_tag: str) -> Any:
        return self.gc.get(f"/annotation/item/{label_id}")

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

    def list_images(self) -> List[str]:
        data = self.gc.get("item", parameters={"folderId": self.folder, "limit": 0})
        return [d["_id"] for d in data if d.get("largeImage")]

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
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    api_url = "http://0.0.0.0:8080/api/v1"
    folder = "621e94e2b6881a7a4bef5170"
    annotation_groups = None
    asset_store_path = "/localhome/sachi/Projects/digital_slide_archive/devops/dsa/assetstore"
    api_key = "OJDE9hjuOIS6R8oEqhnVYHUpRpk18NfJABMt36dJ"

    # api_url = "https://demo.kitware.com/histomicstk/api/v1"
    # folder = "5bbdeba3e629140048d017bb"
    # annotation_groups = ["mostly_tumor"]
    # asset_store_path = None
    # api_key = None

    ds = DSADatastore(api_url, folder, api_key, annotation_groups, asset_store_path)

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

    label_id = labeled_images[0]
    label_tag = "FINAL"
    print(f"Label Info: {json.dumps(ds.get_label_info(label_id, label_tag), indent=2)}")
    print(f"Label URI: {ds.get_label_uri(label_id, label_tag)}")

    print(f"Dataset for Training: \n{json.dumps(ds.datalist(), indent=2)}")

    # http://0.0.0.0:8080/api/v1/item/621e9513b6881a7a4bef517d/tiles/region?left=7102&top=15020&regionWidth=1730&regionHeight=981&units=base_pixels&encoding=PNG
    # http://0.0.0.0:8080/api/v1/item/621e9513b6881a7a4bef517d/tiles/region
    # parameters = {
    #     "left": 6674,
    #     "top": 22449,
    #     "regionWidth": 1038,
    #     "regionHeight": 616,
    #     "units": "base_pixels",
    #     "exact": False,
    #     "encoding": "PNG",
    # }


if __name__ == "__main__":
    main()
