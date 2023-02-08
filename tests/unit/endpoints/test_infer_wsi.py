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

from .context import BasicEndpointV4TestSuite


class EndPointWSIInfer(BasicEndpointV4TestSuite):
    def test_segmentation_asap(self):
        if not torch.cuda.is_available():
            return

        model = "segmentation_nuclei"
        image = "JP2K-33003-1"
        wsi = {"level": 0, "size": [2000, 2000], "pos": [2000, 4000]}

        response = self.client.post(f"/infer/wsi/{model}?image={image}&output=asap", json=wsi)
        assert response.status_code == 200
        time.sleep(1)

    def test_segmentation_dsa(self):
        if not torch.cuda.is_available():
            return

        model = "segmentation_nuclei"
        image = "JP2K-33003-1"
        wsi = {"level": 0, "size": [2000, 2000], "pos": [2000, 4000], "max_workers": 1}

        response = self.client.post(f"/infer/wsi/{model}?image={image}&output=dsa", json=wsi)
        assert response.status_code == 200
        time.sleep(1)

    def test_segmentation_json(self):
        if not torch.cuda.is_available():
            return

        model = "segmentation_nuclei"
        image = "JP2K-33003-1"
        wsi = {"level": 0, "size": [2000, 2000], "pos": [2000, 4000]}

        response = self.client.post(f"/infer/wsi/{model}?image={image}&output=json", json=wsi)
        assert response.status_code == 200
        time.sleep(1)


if __name__ == "__main__":
    unittest.main()
