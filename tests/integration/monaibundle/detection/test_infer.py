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
    def test_lung_nodule_detection(self):
        if not torch.cuda.is_available():
            return

        model = "lung_nodule_ct_detection_v0.5.0"
        image = "1.3.6.1.4.1.14519.5.2.1.6279.6001.188385286346390202873004762827"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
