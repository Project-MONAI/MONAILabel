# Copyright 2020 - 2021 MONAI Consortium
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

from .context import BasicEndpointV2TestSuite


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
            "name": "net_test_01",
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
            "name": "net_test_01",
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


if __name__ == "__main__":
    unittest.main()
