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


class EndPointBatchInfer(unittest.TestCase):
    def test_batch_segmentation_spleen(self):
        if not torch.cuda.is_available():
            return

        model = "segmentation_spleen"
        params = {
            "device": "cuda",
            "multi_gpu": True,
            "gpus": "all",
            "logging": "WARNING",
            "save_label": True,
            "label_tag": "original",
            "max_workers": 0,
            "max_batch_size": 3,
        }

        response = requests.post(f"{SERVER_URI}/batch/infer/{model}?images=all&run_sync=true", json=params)
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
