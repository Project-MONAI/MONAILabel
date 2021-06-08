import os
import unittest

from .context import BasicEndpointTestSuite


class EndPointDatastore(BasicEndpointTestSuite):
    def setUp(self) -> None:
        self.image_id = f"la_99{self.rand_id}.nii.gz"

    def test_001_datastore(self):
        response = self.client.get("/datastore/")
        assert response.status_code == 200

        res = response.json()
        total = res["total"]
        assert total > 0

    def test_002_add(self):
        with open(os.path.join(self.studies, "la_003.nii.gz"), "rb") as f:
            response = self.client.put("/datastore/", files={"file": (self.image_id, f)})
            assert response.status_code == 200
        assert self.image_id in self.client.get("/datastore/?output=all").text

    def test_003_save_label(self):
        tag = "test"
        with open(os.path.join(self.studies, "labels", "label_final_la_003.nii.gz"), "rb") as f:
            response = self.client.put(
                f"/datastore/label?image={self.image_id}&tag={tag}", files={"label": (self.image_id, f)}
            )
            assert response.status_code == 200
            assert self.client.get("/datastore/").json()["label_tags"]["test"]

    def test_004_remove(self):
        total = self.client.get("/datastore/").json()["total"]
        response = self.client.delete(f"/datastore/?id={self.image_id}&type=image")
        assert response.status_code == 200

        current = self.client.get("/datastore/").json()["total"]
        assert current == total - 1


if __name__ == "__main__":
    unittest.main()
