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
    def test_segmentation_spleen(self):
        if not torch.cuda.is_available():
            return

        model = "spleen_ct_segmentation_v0.1.0"
        image = "spleen_8"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200

    def test_segmentation(self):
        if not torch.cuda.is_available():
            return

        model = "swin_unetr_btcv_segmentation_v0.1.0"
        image = "spleen_8"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200

    def disabled_test_segmentation_pancreas(self):
        if not torch.cuda.is_available():
            return

        model = "pancreas_ct_dints_segmentation_v0.1.0"
        image = "spleen_8"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200

    def test_deepedit(self):
        if not torch.cuda.is_available():
            return

        model = "spleen_deepedit_annotation_v0.1.0"
        image = "spleen_8"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
