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


class EndPointInfer(unittest.TestCase):
    def test_deepedit_nuclei(self):
        if not torch.cuda.is_available():
            return

        model = "deepedit_nuclei"
        image = "JP2K-33003-1"
        body = {
            "level": 0,
            "location": [2206, 4925],
            "size": [360, 292],
            "tile_size": [2048, 2048],
            "min_poly_area": 30,
            "params": {"foreground": [], "background": []},
        }

        response = requests.post(f"{SERVER_URI}/infer/wsi/{model}?image={image}&output=dsa", json=body)
        assert response.status_code == 200

    def test_segmentation_nuclei(self):
        if not torch.cuda.is_available():
            return

        model = "segmentation_nuclei"
        image = "JP2K-33003-1"
        body = {
            "level": 0,
            "location": [2206, 4925],
            "size": [360, 292],
            "tile_size": [2048, 2048],
            "min_poly_area": 30,
            "params": {"foreground": [], "background": []},
        }

        response = requests.post(f"{SERVER_URI}/infer/wsi/{model}?image={image}&output=asap", json=body)
        assert response.status_code == 200

    def test_nuclick(self):
        if not torch.cuda.is_available():
            return

        model = "nuclick"
        image = "JP2K-33003-1"
        body = {
            "level": 0,
            "location": [2206, 4925],
            "size": [360, 292],
            "tile_size": [2048, 2048],
            "min_poly_area": 30,
            "params": {
                "foreground": [[2427, 4976], [2341, 5033], [2322, 5207], [2305, 5212], [2268, 5182]],
                "background": [],
            },
        }

        response = requests.post(f"{SERVER_URI}/infer/wsi/{model}?image={image}&output=asap", json=body)
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
