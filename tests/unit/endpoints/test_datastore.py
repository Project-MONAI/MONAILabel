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

import os
import unittest

from .context import BasicEndpointTestSuite


class EndPointDatastore(BasicEndpointTestSuite):
    def setUp(self) -> None:
        self.image_id = f"la_99{self.rand_id}"
        self.image_file = f"{self.image_id}.nii.gz"

    def test_001_datastore(self):
        response = self.client.get("/datastore/")
        assert response.status_code == 200

        res = response.json()
        total = res["total"]
        assert total > 0

    def test_002_add(self):
        with open(os.path.join(self.studies, "la_003.nii.gz"), "rb") as f:
            response = self.client.put("/datastore/", files={"file": (self.image_file, f)})
            assert response.status_code == 200
        assert self.image_id in self.client.get("/datastore/?output=all").text

    def test_003_save_label(self):
        tag = "test"
        with open(os.path.join(self.studies, "labels", "final", "la_003.nii.gz"), "rb") as f:
            response = self.client.put(
                f"/datastore/label?image={self.image_id}&tag={tag}", files={"label": (self.image_file, f)}
            )
            assert response.status_code == 200
            assert self.client.get("/datastore/").json()["label_tags"]["test"]

    def test_004_remove(self):
        total = self.client.get("/datastore/").json()["total"]
        response = self.client.delete(f"/datastore/?id={self.image_id}")
        assert response.status_code == 200

        current = self.client.get("/datastore/").json()["total"]
        assert current == total - 1


if __name__ == "__main__":
    unittest.main()
