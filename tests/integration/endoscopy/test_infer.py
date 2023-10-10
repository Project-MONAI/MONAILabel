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
    def test_tooltracking(self):
        if not torch.cuda.is_available():
            return

        model = "tooltracking"
        image = "Video_2_VTS_01_1_Trim_04-40_f600"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200

    @unittest.skip("Bundle needs to be fixed for AsChannelFirstd")
    def test_inbody(self):
        if not torch.cuda.is_available():
            return

        model = "inbody"
        image = "Video_2_VTS_01_1_Trim_04-40_f600"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}&output=json")
        assert response.status_code == 200

    def test_deepedit(self):
        if not torch.cuda.is_available():
            return

        model = "deepedit"
        image = "Video_2_VTS_01_1_Trim_04-40_f600"

        response = requests.post(f"{SERVER_URI}/infer/{model}?image={image}")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
