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

import time
import unittest

import torch

from .context import BasicBundleTestSuite, BasicDetectionBundleTestSuite, BasicEndpointV2TestSuite


class TestEndPointTrain(BasicEndpointV2TestSuite):
    def test_001_train(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "segmentation_spleen",
            "max_epochs": 1,
            "name": "net_test_01",
            "val_split": 0.5,
            "multi_gpu": False,
            "dataset": "CacheDataset",
        }
        response = self.client.post("/train/?run_sync=True", json=params)
        assert response.status_code == 200
        assert response.json()

    def test_002_train(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "segmentation_spleen",
            "max_epochs": 1,
            "name": "net_test_02",
            "val_split": 0.5,
            "multi_gpu": False,
            "dataset": "CacheDataset",
        }
        response = self.client.post("/train/", json=params)

        assert response.status_code == 200
        assert response.json()
        time.sleep(5)

        assert self.client.get("/train/?check_if_running=True").status_code == 200

        while self.client.get("/train/?check_if_running=True").status_code == 200:
            time.sleep(5)

    def test_003_stop(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "segmentation_spleen",
            "max_epochs": 5,
            "name": "net_test_03",
            "multi_gpu": False,
            "dataset": "CacheDataset",
        }
        response = self.client.post("/train/", json=params)
        assert response.status_code == 200

        time.sleep(5)

        response = self.client.delete("/train/")
        assert response.status_code == 200

    def test_004_status(self):
        self.client.get("/train/")

    def test_005_stop(self):
        self.client.delete("/train/")


class TestBundleTrainTask(BasicBundleTestSuite):
    def test_spleen_bundle_train(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "spleen_ct_segmentation",
            "max_epochs": 1,
            "name": "net_test_spleen_bundle_trainer_01",
            "val_split": 0.5,
            "multi_gpu": False,
            "dataset": "CacheDataset",
        }
        response = self.client.post("/train/?run_sync=True", json=params)
        assert response.status_code == 200


class TestDetectionBundleTrainTask(BasicDetectionBundleTestSuite):
    @unittest.skip("Bundle needs to be fixed for EnsureChannelFirstd init Arguments")
    def test_lung_nodule_detection_train(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "lung_nodule_ct_detection",
            "max_epochs": 1,
            "name": "net_test_lung_nodule_detection_trainer_01",
            "val_split": 0.5,
            "multi_gpu": False,
            "dataset": "CacheDataset",
            "run_id": "run",
        }
        response = self.client.post("/train/?run_sync=True", json=params)
        assert response.status_code == 200

    @unittest.skip("Bundle needs to be fixed for EnsureChannelFirstd init Arguments")
    def test_bundle_stop(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "lung_nodule_ct_detection",
            "max_epochs": 2,
            "name": "net_test_bundle_train",
            "multi_gpu": False,
            "dataset": "CacheDataset",
        }
        response = self.client.post("/train/", json=params)
        assert response.status_code == 200

        time.sleep(5)

        response = self.client.delete("/train/")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
