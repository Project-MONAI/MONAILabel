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
import mimetypes
import os.path

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

MONAI_LABEL_URL = "http://10.33.72.12:8000"
ACTIVE_LEARNING_SAMPLES = 20

CVAT_URL = "http://10.33.72.12:8080"
CVAT_USER = "sachi"
CVAT_PASSWORD = "sachi"

CVAT_PROJECT = "MONAILabel"
CVAT_TASK = "ActiveLearning_Iteration_1"
CVAT_IMAGE_QUALITY = 70


def get_mime_type(file):
    m_type = mimetypes.guess_type(file, strict=False)
    if m_type is None or m_type[0] is None:
        m_type = "application/octet-stream"
    else:
        m_type = m_type[0]
    return m_type


def main():
    auth = HTTPBasicAuth(CVAT_USER, CVAT_PASSWORD)
    projects = requests.get(f"{CVAT_URL}/api/projects", auth=auth).json()
    print(projects)

    project_id = None
    for project in projects["results"]:
        if project["name"] == CVAT_PROJECT:
            project_id = project["id"]
            break

    if project_id is None:
        body = {"name": "MONAILabel", "labels": [{"name": "Tool", "attributes": [], "color": "#66ff66"}]}
        project = requests.post(f"{CVAT_URL}/api/projects", auth=auth, json=body).json()
        print(project)
        project_id = project["id"]

    print(f"Using Project ID: {project_id}")

    tasks = requests.get(f"{CVAT_URL}/api/tasks", auth=auth).json()
    print(tasks)

    task_id = None
    for task in tasks["results"]:
        if task["name"] == CVAT_TASK:
            task_id = task["id"]
            break

    if task_id is None:
        body = {"name": "ActiveLearning_Iteration_1", "labels": [], "project_id": project_id, "subset": "Train"}
        task = requests.post(f"{CVAT_URL}/api/tasks", auth=auth, json=body).json()
        print(task)
        task_id = task["id"]

    print(f"Using Task ID: {task_id}")

    file_list = [
        ("image_quality", (None, f"{CVAT_IMAGE_QUALITY}")),
    ]
    for i in range(ACTIVE_LEARNING_SAMPLES):
        next_sample = requests.post(f"{MONAI_LABEL_URL}/activelearning/tooltracking_epistemic").json()
        image = next_sample['path']
        print(f"Selected Image to upload to CVAT: {image}")
        file_list.append((f"client_files[{i}]", (os.path.basename(image), open(image, 'rb'), get_mime_type(image))))

    r = requests.post(f"{CVAT_URL}/api/tasks/{task_id}/data", files=file_list, auth=auth).json()
    print(r)


if __name__ == "__main__":
    main()
