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
import unittest

import requests
import torch

from tests.integration import SERVER_URI


class EndPointSession(unittest.TestCase):
    def test_segmentation_spleen_trainer(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "spleen_ct_segmentation_v0.1.0",
            "max_epochs": 1,
            "name": "net_test_spleen_trainer_01",
            "val_split": 0.5,
            "multi_gpu": False,
        }

        response = requests.post(f"{SERVER_URI}/train/spleen_ct_segmentation_v0.1.0?run_sync=True", json=params)
        assert response.status_code == 200

    def test_multiorgan_segmentation_trainer(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "swin_unetr_btcv_segmentation_v0.1.0",
            "max_epochs": 1,
            "name": "net_test_multiorgan_trainer_01",
            "val_split": 0.5,
            "multi_gpu": False,
        }

        response = requests.post(f"{SERVER_URI}/train/swin_unetr_btcv_segmentation_v0.1.0?run_sync=True", json=params)
        assert response.status_code == 200

    def test_deepedit(self):
        if not torch.cuda.is_available():
            return

        params = {
            "model": "spleen_deepedit_annotation_v0.1.0",
            "max_epochs": 1,
            "name": "net_test_deepedit_trainer_01",
            "val_split": 0.5,
            "multi_gpu": False,
        }
        response = requests.post(f"{SERVER_URI}/train/spleen_deepedit_annotation_v0.1.0?run_sync=True", json=params)
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
