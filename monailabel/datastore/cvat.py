import logging
import os
import shutil
import tempfile
import time

import requests
from PIL import Image
from requests.auth import HTTPBasicAuth

from monailabel.datastore.local import LocalDatastore
from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.utils.others.generic import get_mime_type

logger = logging.getLogger(__name__)


class CVATDatastore(LocalDatastore):
    def __init__(self, datastore_path, api_url, username=None, password=None, project="MONAILabel", **kwargs):
        self.api_url = api_url.rstrip("/").strip()
        self.auth = HTTPBasicAuth(username, password) if username else None
        self.project = project

        logger.info(f"CVAT:: API URL: {api_url}")
        logger.info(f"CVAT:: UserName: {username}")
        logger.info(f"CVAT:: Password: {'*' * len(password) if password else ''}")
        logger.info(f"CVAT:: Project: {project}")

        super().__init__(datastore_path=datastore_path, **kwargs)

    def name(self) -> str:
        return "CVAT+Local Datastore"

    def description(self) -> str:
        return "CVAT+Local Datastore"

    def get_cvat_project_id(self):
        projects = requests.get(f"{self.api_url}/api/projects", auth=self.auth).json()
        logger.info(projects)

        project_id = None
        for project in projects["results"]:
            if project["name"] == self.project:
                project_id = project["id"]
                break

        if project_id is None:
            body = {"name": "MONAILabel", "labels": [{"name": "Tool", "attributes": [], "color": "#66ff66"}]}
            project = requests.post(f"{self.api_url}/api/projects", auth=self.auth, json=body).json()
            print(project)
            project_id = project["id"]

        logger.info(f"Using Project ID: {project_id}")
        return project_id

    def get_cvat_task_id(self, project_id, task_name):
        tasks = requests.get(f"{self.api_url}/api/tasks", auth=self.auth).json()
        logger.info(tasks)

        task_id = None
        for task in tasks["results"]:
            if task["name"] == task_name:
                task_id = task["id"]
                break

        if task_id is None:
            body = {"name": task_name, "labels": [], "project_id": project_id, "subset": "Train"}
            task = requests.post(f"{self.api_url}/api/tasks", auth=self.auth, json=body).json()
            logger.info(task)
            task_id = task["id"]

        logger.info(f"Using Task ID: {task_id}")
        return project_id, task_id

    def upload_to_cvat(self, samples, task_name="ActiveLearning_Iteration_1", quality=70):
        project_id = self.get_cvat_project_id()
        task_id = self.get_cvat_task_id(project_id, task_name)

        file_list = [("image_quality", (None, f"{quality}"))]
        for i, image in enumerate(samples):
            print(f"Selected Image to upload to CVAT: {image}")
            file_list.append((f"client_files[{i}]", (os.path.basename(image), open(image, "rb"), get_mime_type(image))))

        r = requests.post(f"{self.api_url}/api/tasks/{task_id}/data", files=file_list, auth=self.auth).json()
        logger.info(r)

    def download_from_cvat(self, task_name="ActiveLearning_Iteration_1", max_retry_count=5, wait_time=10):
        project_id = self.get_cvat_project_id()
        task_id = self.get_cvat_task_id(project_id, task_name)

        download_url = f"{self.api_url}/api/tasks/{task_id}/annotations?action=download&format=Segmentation+mask+1.1"
        tmp_folder = tempfile.TemporaryDirectory().name
        os.makedirs(tmp_folder, exist_ok=True)

        tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip").name
        retry_count = 0
        for retry in range(max_retry_count):
            try:
                r = requests.get(download_url, allow_redirects=True, auth=self.auth)
                time.sleep(wait_time)

                open(tmp_zip, "wb").write(r.content)
                shutil.unpack_archive(tmp_zip, tmp_folder)

                segmentations_dir = os.path.join(tmp_folder, "SegmentationClass")
                final_labels = self._datastore.label_path(DefaultLabelTag.FINAL)
                for f in os.listdir(segmentations_dir):
                    label = os.path.join(segmentations_dir, f)
                    if os.path.isfile(label) and label.endswith(".png"):
                        os.makedirs(final_labels, exist_ok=True)

                        dest = os.path.join(final_labels, f)
                        Image.open(label).convert("L").point(lambda x: 0 if x < 128 else 255, "1").save(dest)
                        logger.info(f"Copy Final Label: {label} to {dest}")
                break
            except Exception as e:
                logger.exception(e)
                logger.error(f"{retry} => Failed to download...")
            retry_count = retry_count + 1
