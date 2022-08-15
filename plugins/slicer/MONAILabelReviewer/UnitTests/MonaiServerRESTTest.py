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
import sys
import time
import unittest

import requests
from mockserver_friendly import MockServerFriendlyClient, json_equals, request, response, times
from python_on_whales import docker

sys.path.append("..")
from MONAILabelReviewerLib.MonaiServerREST import MonaiServerREST


class MonaiServerRESTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.info("Start test sets of MonaiServerREST-Test-class")
        cls.MOCK_SERVER_URL = "http://localhost:1080"
        cls.strJson = cls.loadJsonStr("test_json_datastore_v2.json")
        cls.mockServerContainer = docker.run(
            "jamesdbloom/mockserver:mockserver-5.4.1", publish=[(1080, 1080)], detach=True
        )
        if cls.mockServerContainer.state.running is True:
            logging.info(f"Mock server started successfully. Listening on port {1080}")
        else:
            logging.info("Starting mockserver failed")
        time.sleep(8)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mockServerContainer.stop()
        if cls.mockServerContainer.state.running is False:
            logging.info("Mock server terminated successfully")
        cls.mockServerContainer.remove()
        return super().tearDownClass()

    @classmethod
    def loadJsonStr(cls, fileName: str) -> str:
        with open(os.path.join(sys.path[0], "TestDataSet/" + fileName)) as f:
            data = json.dumps(json.load(f))
        return data

    def test_requestDataStoreInfo(cls):

        client = MockServerFriendlyClient(cls.MOCK_SERVER_URL)
        urlToBeTested = "http://localhost:1080/datastore?output=all"
        client.expect(
            request(method="GET", path="/datastore", querystring={"output": "all"}, body=json_equals(cls.strJson)),
            response(code=200),
            times(1),
        )

        result = requests.get(
            cls.MOCK_SERVER_URL + "/datastore",
            params={"output": "all"},
            json=cls.strJson,
            headers={"Content-Type": "application/json"},
        )

        cls.assertEqual(200, result.status_code)
        cls.assertEqual(urlToBeTested, result.url)
        client.reset()

    def test_request_for_image_with_segmentation_meta_data(cls):
        client = MockServerFriendlyClient(cls.MOCK_SERVER_URL)

        urlToBeTested = cls.MOCK_SERVER_URL + "/datastore/updatelabelinfo?label=6662775"
        body = {
            "segmentationMeta": {
                "status": "approved",
                "approvedBy": "",
                "level": "hard",
                "comment": "",
                "editTime": "Fri May 27 07:41:08 2022",
            }
        }

        client.expect(
            request(
                method="PUT",
                path="/datastore/updatelabelinfo",
                querystring={"label": "6662775"},
                headers={"content-Type": "application/json"},
                body=json_equals(body),
            ),
            response(code=200),
            times(1),
        )

        result = requests.put(
            cls.MOCK_SERVER_URL + "/datastore/updatelabelinfo",
            params={"label": "6662775"},
            json=body,
            headers={"content-Type": "application/json"},
        )

        cls.assertEqual(200, result.status_code)
        cls.assertEqual(urlToBeTested, result.url)
        client.reset()

    def test_updateLabelInfo(cls):
        client = MockServerFriendlyClient(cls.MOCK_SERVER_URL)

        urlToBeTested = "{}/datastore/updatelabelinfo?label={}&tag={}".format(
            cls.MOCK_SERVER_URL, str(6662775), "final"
        )
        body = {
            "segmentationMeta": {
                "status": "approved",
                "approvedBy": "",
                "level": "hard",
                "comment": "",
                "editTime": "Fri May 27 07:41:08 2022",
            }
        }

        client.expect(
            request(
                method="PUT",
                path="/datastore/updatelabelinfo",
                querystring={"label": "6662775", "tag": "final"},
                headers={"content-Type": "application/json"},
                body=json_equals(body),
            ),
            response(code=200),
            times(1),
        )

        result = requests.put(
            cls.MOCK_SERVER_URL + "/datastore/updatelabelinfo",
            params={"label": "6662775", "tag": "final"},
            json=body,
            headers={"content-Type": "application/json"},
        )
        cls.assertEqual(200, result.status_code)

        cls.assertEqual(urlToBeTested, result.url)
        client.reset()

    def test_saveLabel(cls):
        client = MockServerFriendlyClient(cls.MOCK_SERVER_URL)
        urlToBeTested = "{}/datastore/label?image={}&tag={}".format(cls.MOCK_SERVER_URL, 6662775, "version_1")
        body = {
            "params": {
                "label_info": [
                    {"name": "Lung", "idx": 1},
                    {"name": "Heart", "idx": 2},
                    {"name": "Trachea", "idx": 3},
                    {"name": "Mediastinum", "idx": 4},
                    {"name": "Clavicle", "idx": 5},
                ],
                "segmentationMeta": {
                    "status": "approved",
                    "approvedBy": "Approver",
                    "level": "hard",
                    "comment": "the_comment",
                    "editTime": 1660488836,
                },
            }
        }

        client.expect(
            request(
                method="PUT",
                path="/datastore/label",
                querystring={"image": "6662775", "tag": "version_1"},
                headers={"content-Type": "application/json"},
                body=json_equals(body),
            ),
            response(code=200),
            times(1),
        )

        result = requests.put(
            cls.MOCK_SERVER_URL + "/datastore/label",
            params={"image": "6662775", "tag": "version_1"},
            json=body,
            headers={"content-Type": "application/json"},
        )

        cls.assertEqual(200, result.status_code)
        cls.assertEqual(urlToBeTested, result.url)
        client.reset()

    def test_deleteLabelByVersionTag(cls):
        client = MockServerFriendlyClient(cls.MOCK_SERVER_URL)

        urlToBeTested = "{}/datastore/label?id={}&tag={}".format(cls.MOCK_SERVER_URL, "6662775", "final")

        client.expect(
            request(method="DELETE", path="/datastore/label", querystring={"id": "6662775", "tag": "final"}),
            response(code=200),
            times(1),
        )

        result = requests.delete(
            cls.MOCK_SERVER_URL + "/datastore/label",
            params={"id": "6662775", "tag": "final"},
        )
        cls.assertEqual(200, result.status_code)

        cls.assertEqual(urlToBeTested, result.url)
        client.reset()

    def test_checkServerConnection(cls):
        client = MockServerFriendlyClient(cls.MOCK_SERVER_URL)
        client.expect(request(method="GET"), response(code=200), times(1))

        result = requests.get(cls.MOCK_SERVER_URL)

        cls.assertEqual(200, result.status_code)
        cls.assertEqual(cls.MOCK_SERVER_URL + "/", result.url)

    def test_requestSegmentation(cls):
        urlToBeTested = cls.MOCK_SERVER_URL + "/datastore/label?label=6662775&tag=final"
        client = MockServerFriendlyClient(cls.MOCK_SERVER_URL)

        client.expect(
            request(method="GET", path="/datastore/label", querystring={"label": "6662775", "tag": "final"}),
            response(code=200),
            times(1),
        )

        result = requests.get(cls.MOCK_SERVER_URL + "/datastore/label", params={"label": "6662775", "tag": "final"})

        cls.assertEqual(200, result.status_code)
        cls.assertEqual(urlToBeTested, result.url)

    def test_getDicomDownloadUri(cls):
        monaiServerREST = MonaiServerREST(cls.MOCK_SERVER_URL)
        imageId = "6662775"
        url = monaiServerREST.getDicomDownloadUri(imageId)
        cls.assertEqual("http://localhost:1080/datastore/image?image=6662775", url)


if __name__ == "__main__":
    unittest.main()
