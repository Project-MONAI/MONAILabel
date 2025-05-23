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

import json
import logging
import os
import shutil
import tempfile
import time
import urllib.parse

import numpy as np
import requests
from PIL import Image
from requests.auth import HTTPBasicAuth

from monailabel.datastore.local import LocalDatastore
from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.utils.others.generic import get_mime_type

logger = logging.getLogger(__name__)


class CVATDatastore(LocalDatastore):
    def __init__(
        self,
        datastore_path,
        api_url,
        username=None,
        password=None,
        project="MONAILabel",
        task_prefix="ActiveLearning_Iteration",
        image_quality=70,
        labels=None,
        normalize_label=True,
        segment_size=1,
        **kwargs,
    ):
        default_labels = [
            {"name": "Tool", "attributes": [], "color": "#66ff66"},
            {"name": "InBody", "attributes": [], "color": "#ff0000"},
            {"name": "OutBody", "attributes": [], "color": "#0000ff"},
        ]
        labels = labels if labels else default_labels
        labels = json.loads(labels) if isinstance(labels, str) else labels

        self.api_url = api_url.rstrip("/").strip()
        self.auth = HTTPBasicAuth(username, password) if username else None
        self.project = project
        self.task_prefix = task_prefix
        self.image_quality = image_quality
        self.labels = labels
        self.label_map = {l["name"]: idx for idx, l in enumerate(labels, start=1)}
        self.normalize_label = normalize_label
        self.segment_size = segment_size

        logger.info(f"CVAT:: API URL: {api_url}")
        logger.info(f"CVAT:: UserName: {username}")
        logger.info(f"CVAT:: Password: {'*' * len(password) if password else ''}")
        logger.info(f"CVAT:: Project: {project}")
        logger.info(f"CVAT:: Task Prefix: {task_prefix}")
        logger.info(f"CVAT:: Image Quality: {image_quality}")
        logger.info(f"CVAT:: Labels: {labels}")
        logger.info(f"CVAT:: Normalize Label: {normalize_label}")
        logger.info(f"CVAT:: Segment Size: {normalize_label}")

        super().__init__(datastore_path=datastore_path, **kwargs)

        self.done_prefix = "DONE"

    def name(self) -> str:
        return "CVAT+Local Datastore"

    def description(self) -> str:
        return "CVAT+Local Datastore"

    def get_cvat_project_id(self, create):
        projects = requests.get(f"{self.api_url}/api/projects", auth=self.auth).json()
        logger.debug(projects)

        project_id = None
        for project in projects["results"]:
            if project["name"] == self.project:
                project_id = project["id"]
                break

        if create and project_id is None:
            body = {"name": self.project, "labels": self.labels}
            project = requests.post(f"{self.api_url}/api/projects", auth=self.auth, json=body).json()
            logger.info(project)
            project_id = project["id"]

        logger.debug(f"Using Project ID: {project_id}")
        return project_id

    def get_cvat_task_id(self, project_id, create):
        filter = {"and": [{"==": [{"var": "project_id"}, project_id]}]}
        filter = urllib.parse.quote_plus(json.dumps(filter))
        tasks = requests.get(f"{self.api_url}/api/tasks?filter={filter}", auth=self.auth).json()

        task_id = None
        task_name = ""
        for task in tasks["results"]:
            if task["name"].startswith(self.task_prefix):
                task_id = task["id"]
                task_name = task["name"] if task["name"] > task_name else task_name

        # increment to next iteration based on latest done_xxx
        if create:
            if not task_name:
                for task in tasks["results"]:
                    if task["name"].startswith(f"{self.done_prefix}_{self.task_prefix}"):
                        task_name = task["name"] if task["name"] > task_name else task_name

            version = int(task_name.split("_")[-1]) + 1 if task_name else 1
            task_name = f"{self.task_prefix}_{version}"
            logger.info(f"Creating new CVAT Task: {task_name}; project: {self.project}")

            body = {"name": task_name, "labels": [], "project_id": project_id, "subset": "Train"}
            if self.segment_size:
                body["segment_size"] = self.segment_size

            task = requests.post(f"{self.api_url}/api/tasks", auth=self.auth, json=body).json()
            logger.debug(task)
            task_id = task["id"]

        logger.debug(f"Using Task ID: {task_id}; Task Name: {task_name}")
        return task_id, task_name

    def task_status(self):
        """
        Fetches the status of a CVAT task based on the state of its jobs.
        Returns:
        - "completed" if all jobs are completed.
        - "in_progress" if at least one job is not completed.
        - None if an error occurs or the task does not exist.
        """
        # Get the project and task IDs
        project_id = self.get_cvat_project_id(create=False)
        if project_id is None:
            return None

        task_id, _ = self.get_cvat_task_id(project_id, create=False)
        if task_id is None:
            return None

        # Fetch task details
        task_url = f"{self.api_url}/api/tasks/{task_id}"
        task_response = requests.get(task_url, auth=self.auth)

        if task_response.status_code != 200:
            return None  # Task could not be fetched

        task_data = task_response.json()

        # Get the jobs URL from the task details
        jobs_url = task_data.get("jobs", {}).get("url")
        if not jobs_url:
            return None  # No jobs URL found for the task

        # Fetch jobs for the task
        jobs_response = requests.get(jobs_url, auth=self.auth)
        if jobs_response.status_code != 200:
            return None  # Jobs could not be fetched

        # Parse jobs and check their states
        jobs = jobs_response.json().get("results", [])
        if not jobs:
            return None  # No jobs found for the task

        # Check if all jobs have state "completed"
        all_completed = all(job.get("state") == "completed" for job in jobs)

        # Return "completed" if all jobs are completed; otherwise "in_progress"
        return "completed" if all_completed else "in_progress"

    def upload_to_cvat(self, samples):
        project_id = self.get_cvat_project_id(create=True)
        task_id, _ = self.get_cvat_task_id(project_id, create=True)

        file_list = [("image_quality", (None, f"{self.image_quality}"))]
        for i, image in enumerate(samples):
            logger.info(f"Selected Image to upload to CVAT: {image}")
            file_list.append((f"client_files[{i}]", (os.path.basename(image), open(image, "rb"), get_mime_type(image))))

        r = requests.post(f"{self.api_url}/api/tasks/{task_id}/data", files=file_list, auth=self.auth).json()
        logger.info(r)

    def trigger_automation(self, function):
        project_id = self.get_cvat_project_id(create=False)
        if project_id is not None:
            task_id, _ = self.get_cvat_task_id(project_id, create=False)
            if task_id is not None:
                body = {"cleanup": True, "task": task_id, "function": function}
                r = requests.post(f"{self.api_url}/api/lambda/requests?org=", json=body, auth=self.auth).json()
                logger.info(r)

    def _load_labelmap_txt(self, file):
        labelmap = {}
        if os.path.exists(file):
            with open(file) as f:
                for line in f.readlines():
                    if line and not line.startswith("#"):
                        fields = line.split(":")
                        name = fields[0]
                        rgb = tuple(int(c) for c in fields[1].split(","))
                        labelmap[name] = rgb
        return labelmap

    def download_from_cvat(self, max_retry_count=5, retry_wait_time=10):
        status = self.task_status()
        if status != "completed":
            logger.info(f"No Tasks with completed status (current: {status}) to refresh/download the final labels")
            return None

        project_id = self.get_cvat_project_id(create=False)
        task_id, task_name = self.get_cvat_task_id(project_id, create=False)
        logger.info(f"Preparing to download/update final labels from: {project_id} => {task_id} => {task_name}")

        # Step 1: Initiate export process using the new POST endpoint.
        export_url = f"{self.api_url}/api/tasks/{task_id}/dataset/export?format=Segmentation+mask+1.1&location=local&save_images=false"
        try:
            response = requests.post(export_url, auth=self.auth)
            if response.status_code not in [200, 202]:
                logger.error(f"Failed to initiate export process: {response.status_code}, {response.text}")
                return None

            rq_id = response.json().get("rq_id")
            if not rq_id:
                logger.error("Export process did not return a request ID (rq_id).")
                return None
            logger.info(f"Export process initiated successfully with request ID: {rq_id}")
        except Exception as e:
            logger.exception(f"Error while initiating export process: {e}")
            return None

        # Step 2: Poll export status using the new GET endpoint.
        status_url = f"{self.api_url}/api/requests/{rq_id}"
        for _ in range(max_retry_count):
            try:
                status_response = requests.get(status_url, auth=self.auth)
                status_data = status_response.json()
                current_status = status_data.get("status")
                if current_status == "finished":
                    logger.info("Export process completed successfully.")
                    break
                elif current_status == "failed":
                    logger.error(f"Export process failed: {status_data}")
                    return None
                logger.info(f"Export in progress... Retrying in {retry_wait_time} seconds.")
                time.sleep(retry_wait_time)
            except Exception as e:
                logger.exception(f"Error checking export status: {e}")
                time.sleep(retry_wait_time)
        else:
            logger.error("Export process did not complete within the maximum retries.")
            return None

        # Step 3: Retrieve the download URL from the export status.
        result_url = status_data.get("result_url")
        if not result_url:
            logger.error("Export process finished but no result_url was provided.")
            return None

        # Step 4: Download the ZIP file from the result_url.
        tmp_folder = tempfile.TemporaryDirectory().name
        os.makedirs(tmp_folder, exist_ok=True)

        tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip").name
        retry_count = 0
        for retry in range(max_retry_count):
            try:
                logger.info(f"Downloading exported dataset from: {result_url}")
                r = requests.get(result_url, allow_redirects=True, auth=self.auth)
                with open(tmp_zip, "wb") as fp:
                    fp.write(r.content)
                shutil.unpack_archive(tmp_zip, tmp_folder)

                # Process the segmentation files
                segmentations_dir = os.path.join(tmp_folder, "SegmentationClass")
                final_labels = self._datastore.label_path(DefaultLabelTag.FINAL)
                for f in os.listdir(segmentations_dir):
                    label = os.path.join(segmentations_dir, f)
                    if os.path.isfile(label) and label.endswith(".png"):
                        os.makedirs(final_labels, exist_ok=True)
                        dest = os.path.join(final_labels, f)
                        if self.normalize_label:
                            img = np.array(Image.open(label))
                            mask = np.zeros_like(img)
                            labelmap = self._load_labelmap_txt(os.path.join(tmp_folder, "labelmap.txt"))
                            for name, color in labelmap.items():
                                if name in self.label_map:
                                    idx = self.label_map.get(name)
                                    mask[np.all(img == color, axis=-1)] = idx
                            Image.fromarray(mask[:, :, 0]).save(dest)
                            logger.info(f"Copied Final Label: {label} to {dest}; unique: {np.unique(mask)}")
                        else:
                            Image.open(label).save(dest)
                            logger.info(f"Copied Final Label: {label} to {dest}")

                # Rename the task to indicate that labels have been processed.
                patch_url = f"{self.api_url}/api/tasks/{task_id}"
                body = {"name": f"{self.done_prefix}_{task_name}"}
                requests.patch(patch_url, allow_redirects=True, auth=self.auth, json=body)
                return task_name
            except Exception as e:
                if retry_count:
                    logger.exception(e)
                logger.error(f"{retry} => Failed to download...")
            retry_count += 1
        return None


"""
def main():
    from pathlib import Path

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.png", "*.jpg", "*.jpeg", ".xml"]
    settings.MONAI_LABEL_DATASTORE = "cvat"
    settings.MONAI_LABEL_DATASTORE_URL = "http://10.117.19.88:8080"
    settings.MONAI_LABEL_DATASTORE_USERNAME = "sachi"
    settings.MONAI_LABEL_DATASTORE_PASSWORD = "sachi"

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Dataset/picked/all"

    ds = CVATDatastore(
        datastore_path=studies,
        api_url=settings.MONAI_LABEL_DATASTORE_URL,
        username=settings.MONAI_LABEL_DATASTORE_USERNAME,
        password=settings.MONAI_LABEL_DATASTORE_PASSWORD,
        project="MONAILabel",
        task_prefix="ActiveLearning_Iteration",
        image_quality=70,
        labels=None,
        normalize_label=True,
        segment_size=0,
        extensions=settings.MONAI_LABEL_DATASTORE_FILE_EXT,
        auto_reload=settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD,
    )
    ds.download_from_cvat()

    # studies = f"{home}/Dataset/Holoscan/flattened/images"


if __name__ == "__main__":
    main()
"""
