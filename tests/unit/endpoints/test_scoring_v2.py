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

import torch

from .context import BasicBundleV2TestSuite


class EndPointBundleScoring(BasicBundleV2TestSuite):
    # test epistemic_v2 
    def test_bundle_epistemic(self):
        if not torch.cuda.is_available():
            return

        response = self.client.post("/scoring/spleen_ct_segmentation_v0.3.1?run_sync=true")
        assert response.status_code == 200

    def test_status(self):
        self.client.get("/scoring/")

    def test_stop(self):
        self.client.delete("/scoring/")

if __name__ == "__main__":
    unittest.main()
