import time
import unittest

from .context import BasicEndpointTestSuite


class TestEndPointTrain(BasicEndpointTestSuite):
    def test_001_train(self):
        params = {"epochs": 1, "name": "net_test_01", "val_split": 0.5}
        response = self.client.post("/train/", json=params)

        assert response.status_code == 200
        assert response.json()
        time.sleep(5)

        assert self.client.get("/train/?check_if_running=True").status_code == 200

        while self.client.get("/train/?check_if_running=True").status_code == 200:
            time.sleep(5)

    def test_002_stop(self):
        params = {"epochs": 3, "name": "net_test_01"}
        response = self.client.post("/train/", json=params)
        assert response.status_code == 200

        time.sleep(5)

        response = self.client.delete("/train/")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
