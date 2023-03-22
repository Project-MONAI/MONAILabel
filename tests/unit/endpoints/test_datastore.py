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

import os
import time
import unittest

from .context import BasicEndpointTestSuite


class EndPointDatastore(BasicEndpointTestSuite):
    def setUp(self) -> None:
        self.image_id = f"spleen_99{self.rand_id}"
        self.image_file = f"{self.image_id}.nii.gz"
        time.sleep(1)

    def test_001_datastore(self):
        response = self.client.get("/datastore/")
        assert response.status_code == 200

        res = response.json()
        total = res["total"]
        assert total > 0
        time.sleep(1)

    def test_002_add(self):
        with open(os.path.join(self.studies, "spleen_3.nii.gz"), "rb") as f:
            response = self.client.put("/datastore/", files={"file": (self.image_file, f)})
            assert response.status_code == 200
        assert self.image_id in self.client.get("/datastore/?output=all").text
        time.sleep(1)

    def test_003_save_remove_label(self):
        tag = "test"
        with open(os.path.join(self.studies, "labels", "final", "spleen_3.nii.gz"), "rb") as f:
            response = self.client.put(
                f"/datastore/label?image={self.image_id}&tag={tag}", files={"label": (self.image_file, f)}
            )
            assert response.status_code == 200
            assert self.client.get("/datastore/").json()["label_tags"]["test"]
            response = self.client.delete(f"/datastore/label?id={self.image_id}&tag={tag}")

            assert response.status_code == 200
            assert tag not in self.client.get("/datastore/").json()["label_tags"]
        time.sleep(1)

    def test_004_remove_image(self):
        total = self.client.get("/datastore/").json()["total"]
        response = self.client.delete(f"/datastore/?id={self.image_id}")
        assert response.status_code == 200

        current = self.client.get("/datastore/").json()["total"]
        assert current == total - 1
        time.sleep(1)

    def test_005_download_image(self):
        response = self.client.get("/datastore/image?image=spleen_3")
        assert response.status_code == 200
        time.sleep(1)

    def test_006_download_label(self):
        response = self.client.get("/datastore/label?label=spleen_3&tag=final")
        assert response.status_code == 200
        time.sleep(1)

    def test_007_get_image_info(self):
        response = self.client.get("/datastore/image/info?image=spleen_3")
        assert response.status_code == 200
        name = self.client.get("/datastore/image/info?image=spleen_3").json()["name"]
        assert name == "spleen_3.nii.gz"
        time.sleep(1)

    def test_008_get_label_info(self):
        response = self.client.get("/datastore/label/info?label=spleen_3&tag=final")
        name = self.client.get("/datastore/label/info?label=spleen_3&tag=final").json()["name"]
        assert response.status_code == 200
        assert name == "spleen_3.nii.gz"
        time.sleep(1)

    def test_009_update_image_info(self):
        response = self.client.put("/datastore/image/info?image=spleen_3")
        name = self.client.get("/datastore/image/info?image=spleen_3").json()["name"]
        assert response.status_code == 200
        assert name == "spleen_3.nii.gz"
        time.sleep(1)

    def test_0010_update_label_info(self):
        response = self.client.put(
            "/datastore/label/info?label=spleen_3&tag=final",
        )
        assert response.status_code == 200
        time.sleep(1)

    def test_011_download_dataset(self):
        response = self.client.get("/datastore/dataset")
        assert response.status_code == 200
        time.sleep(1)


if __name__ == "__main__":
    unittest.main()
