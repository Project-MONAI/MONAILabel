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

import unittest

import torch

from .context import BasicEndpointTestSuite


class TestEndPointLogs(BasicEndpointTestSuite):
    def test_logs(self):
        response = self.client.get("/logs/")
        assert response.status_code == 200

    def test_logs_all(self):
        response = self.client.get("/logs/?lines=0")
        assert response.status_code == 200

    def test_logs_gpu(self):
        if not torch.cuda.is_available():
            return

        response = self.client.get("/logs/gpu")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
