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
import argparse
import json
import mimetypes
import os.path
import shutil
import tempfile
import time

import requests
from PIL import Image
from requests.auth import HTTPBasicAuth


def get_mime_type(file):
    m_type = mimetypes.guess_type(file, strict=False)
    if m_type is None or m_type[0] is None:
        m_type = "application/octet-stream"
    else:
        m_type = m_type[0]
    return m_type


def main():
    MONAI_LABEL_URL = "http://10.117.20.110:8000"
    MONAI_LABEL_STUDIES_PATH = "/localhome/sachi/Dataset/Holoscan/tiny/images"
    MONAI_LABEL_TOP_ACTIVE_LEARNING_SAMPLES = 20
    MONAI_LABEL_TRAIN_EPOCHS = 10

    CVAT_URL = "http://10.117.20.110:8080"
    CVAT_USER = "sachi"
    CVAT_PASSWORD = "sachi"

    CVAT_PROJECT = "MONAILabel"
    CVAT_TASK = "ActiveLearning_Iteration_1"
    CVAT_IMAGE_QUALITY = 70
    CVAT_DOWNLOAD_WAIT_TIME_SEC = 10
    CVAT_DOWNLOAD_RETRY_COUNT = 5

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        default="upload",
        choices=("upload", "train"),
        help="upload: Upload Tasks to CVAT; train: Download Annotations from CVAT and run Trianing",
    )
    parser.add_argument("--monailabel_url", default=MONAI_LABEL_URL, help="MONAI Label server url")
    parser.add_argument("--monailabel_studies", default=MONAI_LABEL_STUDIES_PATH, help="MONAI Label studies path")
    parser.add_argument(
        "--monailabel_top_samples",
        default=MONAI_LABEL_TOP_ACTIVE_LEARNING_SAMPLES,
        help="How many samples to upload as part of Active Learning Strategy",
    )
    parser.add_argument("--monailabel_epochs", default=MONAI_LABEL_TRAIN_EPOCHS, help="MONAI Label train epochs")

    parser.add_argument("--cvat_url", default=CVAT_URL, help="CVAT server url")
    parser.add_argument("--cvat_user", default=CVAT_USER, help="CVAT user name for authentication")
    parser.add_argument("--cvat_password", default=CVAT_PASSWORD, help="CVAT password for authentication")

    parser.add_argument("--cvat_project_name", default=CVAT_PROJECT, help="CVAT Project to be used for creating task")
    parser.add_argument(
        "--cvat_task_name", default=CVAT_TASK, help="CVAT Task name under which images shall be uploaded"
    )
    parser.add_argument("--cvat_image_quality", default=CVAT_IMAGE_QUALITY, help="Image quality")
    parser.add_argument(
        "--cvat_download_wait", default=CVAT_DOWNLOAD_WAIT_TIME_SEC, help="CVAT Download Annotation wait time"
    )
    parser.add_argument(
        "--cvat_download_retires", default=CVAT_DOWNLOAD_RETRY_COUNT, help="Retry count to download CVAT Annotations"
    )

    args = parser.parse_args()
    action = args.action
    MONAI_LABEL_URL = args.monailabel_url
    MONAI_LABEL_STUDIES_PATH = args.monailabel_studies
    MONAI_LABEL_TOP_ACTIVE_LEARNING_SAMPLES = args.monailabel_top_samples
    MONAI_LABEL_TRAIN_EPOCHS = args.monailabel_epochs

    CVAT_URL = args.cvat_url
    CVAT_USER = args.cvat_user
    CVAT_PASSWORD = args.cvat_password

    CVAT_PROJECT = args.cvat_project_name
    CVAT_TASK = args.cvat_task_name
    CVAT_IMAGE_QUALITY = args.cvat_image_quality
    CVAT_DOWNLOAD_WAIT_TIME_SEC = args.cvat_download_wait
    CVAT_DOWNLOAD_RETRY_COUNT = args.cvat_download_retires

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

    # Upload Tasks to CVAT for annotations
    if action == "upload":
        file_list = [
            ("image_quality", (None, f"{CVAT_IMAGE_QUALITY}")),
        ]
        for i in range(MONAI_LABEL_TOP_ACTIVE_LEARNING_SAMPLES):
            next_sample = requests.post(f"{MONAI_LABEL_URL}/activelearning/tooltracking_epistemic").json()
            image = next_sample["path"]
            print(f"Selected Image to upload to CVAT: {image}")
            file_list.append((f"client_files[{i}]", (os.path.basename(image), open(image, "rb"), get_mime_type(image))))

        r = requests.post(f"{CVAT_URL}/api/tasks/{task_id}/data", files=file_list, auth=auth).json()
        print(r)
        return

    # Training...
    download_url = f"{CVAT_URL}/api/tasks/{task_id}/annotations?action=download&format=Segmentation+mask+1.1"
    tmp_folder = tempfile.TemporaryDirectory().name
    os.makedirs(tmp_folder, exist_ok=True)

    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip").name
    retry_count = 0
    for retry in range(CVAT_DOWNLOAD_RETRY_COUNT):
        try:
            r = requests.get(download_url, allow_redirects=True, auth=auth)
            time.sleep(CVAT_DOWNLOAD_WAIT_TIME_SEC)

            open(tmp_zip, "wb").write(r.content)
            shutil.unpack_archive(tmp_zip, tmp_folder)

            segmentations_dir = os.path.join(tmp_folder, "SegmentationClass")
            final_labels = os.path.join(MONAI_LABEL_STUDIES_PATH, "labels", "final")
            for f in os.listdir(segmentations_dir):
                label = os.path.join(segmentations_dir, f)
                if os.path.isfile(label) and label.endswith(".png"):
                    os.makedirs(final_labels, exist_ok=True)

                    dest = os.path.join(final_labels, f.rstrip(".png") + ".jpg")
                    Image.open(label).convert("L").point(lambda x: 0 if x < 128 else 255, "1").save(dest)
                    print(f"Copy Final Label: {label} to {dest}")
            break
        except Exception as e:
            print(e)
            print(f"{retry} => Failed to download...")
        retry_count = retry_count + 1

    if retry_count == CVAT_DOWNLOAD_RETRY_COUNT:
        print("Failed:: Go to browser and manually download + import mask to MONAILabel Server")
    else:
        request = {
            "model": "tooltracking",
            "max_epochs": MONAI_LABEL_TRAIN_EPOCHS,
            "dataset": "CacheDataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 4,
            "val_batch_size": 2,
            "multi_gpu": False,
            "val_split": 0.1,
        }

        print("")
        print("Triggering Training job over MONAILabel server....")
        r = requests.post(f"{MONAI_LABEL_URL}/train/tooltracking?run_sync=true", json=request)
        print("Result Metrics:")
        print(json.dumps(r.json(), indent=2))

    shutil.rmtree(tmp_folder, ignore_errors=True)
    if os.path.exists(tmp_zip):
        os.unlink(tmp_zip)


if __name__ == "__main__":
    main()
