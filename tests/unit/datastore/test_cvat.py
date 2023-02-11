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
import os
import unittest
from unittest.mock import patch

from monailabel.datastore.cvat import CVATDatastore


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

    url = args[0]
    if "/api/projects" in url:
        return MockResponse({"results": [{"id": 1, "name": "project1"}]}, 200)
    elif "/api/tasks/1" in url:
        return MockResponse({"status": "completed"}, 200)
    elif "/api/tasks" in url:
        return MockResponse({"results": [{"id": 1, "name": "task1", "project_id": 1}]}, 200)

    return MockResponse(None, 404)


def mocked_requests_post(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

    url = args[0]
    if "/api/projects" in url:
        return MockResponse({"id": 1, "name": "project1"}, 200)
    elif "/api/tasks" in url:
        return MockResponse({"id": 1, "name": "task1", "status": "completed"}, 200)

    return MockResponse(None, 404)


class TestCVAT(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    @patch("requests.get", side_effect=mocked_requests_get)
    @patch("requests.post", side_effect=mocked_requests_post)
    def test_fetch(self, mock_get, mock_post):
        ds = CVATDatastore(
            datastore_path=os.path.join(TestCVAT.base_dir, "data", "endoscopy"),
            api_url="http://test.cvat.com",
            username="abc",
            password="pwd",
            project="MONAILabel",
            task_prefix="ActiveLearning_Iteration",
            image_quality=70,
            labels=None,
            normalize_label=True,
            segment_size=0,
            extensions=["*.png", "*.jpg", "*.jpeg", ".xml"],
            auto_reload=False,
        )

        ds.name()
        ds.description()
        ds.get_cvat_project_id(create=True)

        project_id = ds.get_cvat_project_id(create=True)
        ds.get_cvat_task_id(project_id=project_id, create=True)
        ds.task_status()
        ds.upload_to_cvat([ds.get_image_uri("frame001"), ds.get_image_uri("frame002")])
        ds.trigger_automation(function=None)

        try:
            ds.project = "project1"
            ds.task_prefix = "task"
            ds.download_from_cvat(max_retry_count=2, retry_wait_time=1)
        except:
            pass
